"""Frozen no-redraw contract for ambiguous stochastic transport attempts.

An HTTP connection failure or timeout cannot prove that a seeded generation was never
accepted by the serving process.  Reissuing that coordinate would therefore risk
replacement sampling.  Schema-5 journals one durable request intent before transport
and turns every ambiguous attempt into a terminal, explicitly reported censor.

Deterministic calibration probes are intentionally outside this protocol: they do not
sample a trajectory and may retain their existing bounded retry behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping


TRANSPORT_CENSOR_PROTOCOL_VERSION = 1
STOCHASTIC_CHAT_SDK_MAX_RETRIES = 0
STOCHASTIC_CHAT_CREATE_CALLS_PER_ATTEMPT = 1
TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES = 512
TRANSPORT_CENSOR_ERROR_REDACTION_POLICY_VERSION = 1
TRANSPORT_CENSOR_ERROR_REDACTION_MARKER = (
    "<redacted sensitive exception message>"
)
TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN = (
    r"(?i)(?:authorization|proxy-authorization|api[_ -]?key|access[_ -]?token|"
    r"refresh[_ -]?token|password|passwd|secret|bearer\s|basic\s|"
    r"sk-[A-Za-z0-9_-]{8,}|://[^\s/@:]+:[^\s/@]+@)"
)
TRANSPORT_CENSOR_CLASS_CONNECTION = "api_connection_ambiguous"
TRANSPORT_CENSOR_CLASS_TIMEOUT = "api_timeout_ambiguous"
TRANSPORT_CENSOR_CLASS_STATUS = "api_status_ambiguous"
TRANSPORT_CENSOR_CLASS_INTERRUPTED = "pending_attempt_interrupted"
TRANSPORT_CENSOR_CLASS_PRODUCER_EXCEPTION = "producer_exception_ambiguous"
TRANSPORT_CENSOR_API_STATUS_ERROR_TYPES = frozenset(
    {
        "openai.APIStatusError",
        "openai.AuthenticationError",
        "openai.BadRequestError",
        "openai.ConflictError",
        "openai.InternalServerError",
        "openai.NotFoundError",
        "openai.PermissionDeniedError",
        "openai.RateLimitError",
        "openai.UnprocessableEntityError",
    }
)
TRANSPORT_CENSOR_CLASSIFICATIONS = frozenset(
    {
        TRANSPORT_CENSOR_CLASS_CONNECTION,
        TRANSPORT_CENSOR_CLASS_TIMEOUT,
        TRANSPORT_CENSOR_CLASS_STATUS,
        TRANSPORT_CENSOR_CLASS_INTERRUPTED,
        TRANSPORT_CENSOR_CLASS_PRODUCER_EXCEPTION,
    }
)
_TRANSPORT_CENSOR_PROTOCOL_SPEC = {
    "version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
    "scope": "schema-5 stochastic topology and self-consistency coordinates",
    "intent_boundary": (
        "fsync exact coordinate request, attempt ID, endpoint generation, and start "
        "timestamp before invoking the stochastic producer"
    ),
    "single_attempt_transport": {
        "openai_sdk_max_retries": STOCHASTIC_CHAT_SDK_MAX_RETRIES,
        "chat_completions_create_calls_per_journaled_attempt": (
            STOCHASTIC_CHAT_CREATE_CALLS_PER_ATTEMPT
        ),
        "stochastic_chat_backoff_decorator": False,
    },
    "ambiguous_failures": {
        "APIConnectionError": TRANSPORT_CENSOR_CLASS_CONNECTION,
        "APITimeoutError": TRANSPORT_CENSOR_CLASS_TIMEOUT,
        "APIStatusError_and_subclasses": TRANSPORT_CENSOR_CLASS_STATUS,
        "process_or_interpreter_loss_with_pending_intent": (
            TRANSPORT_CENSOR_CLASS_INTERRUPTED
        ),
        "any_other_exception_after_durable_stochastic_intent": (
            TRANSPORT_CENSOR_CLASS_PRODUCER_EXCEPTION
        ),
        "registered_api_status_error_types": sorted(
            TRANSPORT_CENSOR_API_STATUS_ERROR_TYPES
        ),
    },
    "retention": (
        "one trusted transport_censored outcome per ambiguous attempt; preserve exact "
        "seed/coordinate/request hash/endpoint/attempt/timestamps/classification and "
        "a secret-safe bounded exception envelope; retain a full-message digest only "
        "for non-sensitive text and explicitly withhold it for redacted text"
    ),
    "exception_envelope": {
        "utf8_encoding_errors": "replace",
        "readable_message_max_utf8_bytes": (
            TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES
        ),
        "fully_qualified_exception_type": True,
        "redaction_policy_version": (
            TRANSPORT_CENSOR_ERROR_REDACTION_POLICY_VERSION
        ),
        "sensitive_message_regex": TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN,
        "sensitive_message_regex_sha256": hashlib.sha256(
            TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN.encode("utf-8")
        ).hexdigest(),
        "redaction_marker": TRANSPORT_CENSOR_ERROR_REDACTION_MARKER,
        "non_sensitive_digest_state": "full_message",
        "non_sensitive_digest": "sha256(full UTF-8 message bytes)",
        "sensitive_digest_state": "withheld_sensitive",
        "sensitive_digest": None,
        "process_restart": {
            "error_detail_state": "process_restart",
            "all_exception_envelope_fields": None,
        },
    },
    "replacement_sampling": "forbidden",
    "calibration_probe_policy": (
        "deterministic forced-option calibration probes remain retryable and are not "
        "stochastic coordinate attempts"
    ),
}
TRANSPORT_CENSOR_PROTOCOL_HASH = hashlib.sha256(
    json.dumps(
        _TRANSPORT_CENSOR_PROTOCOL_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

ATTEMPT_FIELDS = frozenset(
    {
        "attempt_id",
        "coordinate_key",
        "request_sha256",
        "endpoint_generation",
        "started_at",
    }
)
TRANSPORT_CENSOR_FIELDS = frozenset(
    {
        "transport_censor_protocol_version",
        "transport_censor_protocol_hash",
        "error_classification",
        "error_detail_state",
        "error_type",
        "error_message",
        "error_message_digest_state",
        "error_message_sha256",
        "error_message_utf8_bytes",
        "error_message_truncated",
        "error_message_redacted",
        "sampling_attempt_count",
        "qid",
        "agent_id",
        "round",
        "generation_role",
        "sample_index",
        "seed",
        "coordinate_key",
        "request_sha256",
        "attempt_id",
        "endpoint_generation",
        "attempt_started_at",
        "censored_at",
    }
)
_HEX_32_RE = re.compile(r"[0-9a-f]{32}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_QUALIFIED_TYPE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)
_SENSITIVE_MESSAGE_RE = re.compile(TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN)


class TransportCensorError(RuntimeError):
    """One durable ambiguous stochastic attempt, replayed without another draw."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        validated = validate_transport_censor(payload)
        self.payload = validated
        super().__init__(
            "stochastic coordinate was transport-censored without replacement: "
            f"coordinate={validated['coordinate_key']}, "
            f"attempt={validated['attempt_id']}, "
            f"classification={validated['error_classification']}"
        )

    def to_transport_censor(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)


def transport_censor_protocol_spec() -> dict[str, Any]:
    """Return the exact canonical contract covered by the protocol hash."""

    return copy.deepcopy(_TRANSPORT_CENSOR_PROTOCOL_SPEC)


def canonical_sha256(value: Any) -> str:
    """Return the canonical finite-JSON SHA-256 used by attempts and checkpoints."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_timestamp(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{label} must be a finite positive timestamp")
    return float(value)


def _exception_type(error: BaseException) -> str:
    cls = type(error)
    return f"{cls.__module__}.{cls.__qualname__}"


def _safe_error_envelope(error: BaseException | None) -> dict[str, Any]:
    """Return bounded diagnostic text without persisting credentials or headers.

    For non-sensitive messages, the digest and byte count bind the complete original
    ``str(error)`` while the readable excerpt is a UTF-8-safe prefix.  If a secret
    marker is detected, both raw text and its guessable digest are withheld and only a
    fixed redaction marker plus byte count remain.
    """

    if error is None:
        return {
            "error_detail_state": "process_restart",
            "error_type": None,
            "error_message": None,
            "error_message_digest_state": None,
            "error_message_sha256": None,
            "error_message_utf8_bytes": None,
            "error_message_truncated": None,
            "error_message_redacted": None,
        }
    error_type = _exception_type(error)
    raw = str(error).encode("utf-8", errors="replace")
    redacted = _SENSITIVE_MESSAGE_RE.search(raw.decode("utf-8")) is not None
    prefix = raw[:TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES]
    decoded_prefix = prefix.decode("utf-8", errors="ignore")
    truncated = (
        len(raw) > TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES
        or decoded_prefix.encode("utf-8") != prefix
    )
    if redacted:
        excerpt = TRANSPORT_CENSOR_ERROR_REDACTION_MARKER
    else:
        excerpt = decoded_prefix
    return {
        "error_detail_state": "captured",
        "error_type": error_type,
        "error_message": excerpt,
        "error_message_digest_state": (
            "withheld_sensitive" if redacted else "full_message"
        ),
        "error_message_sha256": (
            None if redacted else hashlib.sha256(raw).hexdigest()
        ),
        "error_message_utf8_bytes": len(raw),
        "error_message_truncated": truncated,
        "error_message_redacted": redacted,
    }


def _validate_error_envelope(
    payload: Mapping[str, Any], *, classification: str
) -> None:
    state = payload["error_detail_state"]
    detail_fields = (
        "error_type",
        "error_message",
        "error_message_digest_state",
        "error_message_sha256",
        "error_message_utf8_bytes",
        "error_message_truncated",
        "error_message_redacted",
    )
    if classification == TRANSPORT_CENSOR_CLASS_INTERRUPTED:
        if state != "process_restart" or any(
            payload[field] is not None for field in detail_fields
        ):
            raise ValueError(
                "interrupted transport censor must use the null process-restart "
                "error sentinel"
            )
        return
    if state != "captured":
        raise ValueError("transport exception censor must contain captured error detail")
    error_type = payload["error_type"]
    if (
        not isinstance(error_type, str)
        or _QUALIFIED_TYPE_RE.fullmatch(error_type) is None
    ):
        raise ValueError("transport censor error type must be fully qualified")
    if (
        classification == TRANSPORT_CENSOR_CLASS_CONNECTION
        and error_type != "openai.APIConnectionError"
    ):
        raise ValueError("connection transport censor has the wrong exception type")
    if (
        classification == TRANSPORT_CENSOR_CLASS_TIMEOUT
        and error_type != "openai.APITimeoutError"
    ):
        raise ValueError("timeout transport censor has the wrong exception type")
    if (
        classification == TRANSPORT_CENSOR_CLASS_STATUS
        and error_type not in TRANSPORT_CENSOR_API_STATUS_ERROR_TYPES
    ):
        raise ValueError("status transport censor has the wrong exception type")
    message = payload["error_message"]
    message_bytes = payload["error_message_utf8_bytes"]
    digest_state = payload["error_message_digest_state"]
    message_sha256 = payload["error_message_sha256"]
    truncated = payload["error_message_truncated"]
    redacted = payload["error_message_redacted"]
    if (
        not isinstance(message, str)
        or len(message.encode("utf-8")) > TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES
        or not isinstance(message_bytes, int)
        or isinstance(message_bytes, bool)
        or message_bytes < 0
        or not isinstance(truncated, bool)
        or not isinstance(redacted, bool)
    ):
        raise ValueError("transport censor error message envelope is invalid")
    if message_bytes > TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES and not truncated:
        raise ValueError("transport censor error truncation flag is inconsistent")
    if (
        message_bytes <= TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES
        and truncated
    ):
        raise ValueError("short transport censor message cannot be truncated")
    if redacted:
        if (
            message != TRANSPORT_CENSOR_ERROR_REDACTION_MARKER
            or digest_state != "withheld_sensitive"
            or message_sha256 is not None
        ):
            raise ValueError("transport censor redacted message marker is invalid")
    elif (
        _SENSITIVE_MESSAGE_RE.search(message) is not None
        or digest_state != "full_message"
        or not isinstance(message_sha256, str)
        or _SHA256_RE.fullmatch(message_sha256) is None
        or (
            not truncated
            and (
                len(message.encode("utf-8")) != message_bytes
                or hashlib.sha256(message.encode("utf-8")).hexdigest()
                != message_sha256
            )
        )
    ):
        raise ValueError(
            "transport censor non-sensitive message digest envelope is invalid"
        )


def validate_attempt(
    value: Any,
    *,
    coordinate_key: str | None = None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly validate one pre-transport intent."""

    if not isinstance(value, Mapping) or set(value) != ATTEMPT_FIELDS:
        raise ValueError("transport attempt has the wrong fields")
    attempt = dict(value)
    if (
        not isinstance(attempt["attempt_id"], str)
        or _HEX_32_RE.fullmatch(attempt["attempt_id"]) is None
    ):
        raise ValueError("transport attempt ID must be 32 lowercase hex characters")
    if (
        not isinstance(attempt["coordinate_key"], str)
        or not attempt["coordinate_key"]
        or (
            coordinate_key is not None
            and attempt["coordinate_key"] != coordinate_key
        )
    ):
        raise ValueError("transport attempt coordinate key is invalid")
    if (
        not isinstance(attempt["request_sha256"], str)
        or _SHA256_RE.fullmatch(attempt["request_sha256"]) is None
        or (
            request is not None
            and attempt["request_sha256"] != canonical_sha256(request)
        )
    ):
        raise ValueError("transport attempt request hash is invalid")
    if (
        not isinstance(attempt["endpoint_generation"], str)
        or not attempt["endpoint_generation"]
        or attempt["endpoint_generation"] == "mixed"
    ):
        raise ValueError("transport attempt endpoint generation must be exact")
    _positive_timestamp(attempt["started_at"], "transport attempt started_at")
    return copy.deepcopy(attempt)


def build_transport_censor(
    *,
    request: Mapping[str, Any],
    attempt: Mapping[str, Any],
    classification: str,
    censored_at: float,
    error: BaseException | None,
) -> dict[str, Any]:
    """Materialize a structural censor from one already-durable request intent."""

    validated_attempt = validate_attempt(
        attempt,
        coordinate_key=str(attempt.get("coordinate_key", "")),
        request=request,
    )
    if classification not in TRANSPORT_CENSOR_CLASSIFICATIONS:
        raise ValueError(f"unregistered transport classification {classification!r}")
    if (
        classification == TRANSPORT_CENSOR_CLASS_INTERRUPTED
        and error is not None
    ) or (
        classification != TRANSPORT_CENSOR_CLASS_INTERRUPTED
        and error is None
    ):
        raise ValueError(
            "transport censor exception detail does not match its classification"
        )
    payload = {
        "transport_censor_protocol_version": TRANSPORT_CENSOR_PROTOCOL_VERSION,
        "transport_censor_protocol_hash": TRANSPORT_CENSOR_PROTOCOL_HASH,
        "error_classification": classification,
        **_safe_error_envelope(error),
        "sampling_attempt_count": 1,
        "qid": request.get("qid"),
        "agent_id": request.get("agent_id"),
        "round": request.get("round"),
        "generation_role": request.get("generation_role"),
        "sample_index": request.get("sample_index"),
        "seed": request.get("seed"),
        "coordinate_key": validated_attempt["coordinate_key"],
        "request_sha256": validated_attempt["request_sha256"],
        "attempt_id": validated_attempt["attempt_id"],
        "endpoint_generation": validated_attempt["endpoint_generation"],
        "attempt_started_at": validated_attempt["started_at"],
        "censored_at": censored_at,
    }
    return validate_transport_censor(
        payload,
        request=request,
        attempt=validated_attempt,
    )


def validate_transport_censor(
    value: Any,
    *,
    request: Mapping[str, Any] | None = None,
    attempt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a censor independently of mutable scheduler or serving state."""

    if not isinstance(value, Mapping) or set(value) != TRANSPORT_CENSOR_FIELDS:
        raise ValueError("transport censor has the wrong fields")
    payload = dict(value)
    if (
        payload["transport_censor_protocol_version"]
        != TRANSPORT_CENSOR_PROTOCOL_VERSION
        or payload["transport_censor_protocol_hash"]
        != TRANSPORT_CENSOR_PROTOCOL_HASH
    ):
        raise ValueError("transport censor protocol identity is not registered")
    classification = payload["error_classification"]
    if classification not in TRANSPORT_CENSOR_CLASSIFICATIONS:
        raise ValueError("transport censor classification is not registered")
    _validate_error_envelope(payload, classification=classification)
    if payload["sampling_attempt_count"] != 1:
        raise ValueError("transport censor must represent exactly one sampling attempt")
    started = _positive_timestamp(
        payload["attempt_started_at"], "transport censor attempt_started_at"
    )
    finished = _positive_timestamp(payload["censored_at"], "transport censor censored_at")
    if finished < started:
        raise ValueError("transport censor timestamp precedes its attempt")

    reconstructed_attempt = {
        "attempt_id": payload["attempt_id"],
        "coordinate_key": payload["coordinate_key"],
        "request_sha256": payload["request_sha256"],
        "endpoint_generation": payload["endpoint_generation"],
        "started_at": payload["attempt_started_at"],
    }
    validated_attempt = validate_attempt(
        reconstructed_attempt,
        coordinate_key=(
            None if attempt is None else str(attempt.get("coordinate_key", ""))
        ),
        request=request,
    )
    if attempt is not None and validated_attempt != dict(attempt):
        raise ValueError("transport censor does not match its durable attempt")

    coordinate_fields = (
        "qid",
        "agent_id",
        "round",
        "generation_role",
        "sample_index",
        "seed",
    )
    if request is not None:
        for field in coordinate_fields:
            if payload[field] != request.get(field):
                raise ValueError(
                    f"transport censor {field} does not match its coordinate request"
                )
    else:
        if (
            not isinstance(payload["qid"], str)
            or not payload["qid"]
            or not isinstance(payload["agent_id"], str)
            or not payload["agent_id"]
            or payload["generation_role"] not in {"topology", "self_consistency"}
            or isinstance(payload["round"], bool)
            or not isinstance(payload["round"], int)
            or payload["round"] < 0
            or isinstance(payload["seed"], bool)
            or not isinstance(payload["seed"], int)
            or payload["seed"] < 0
        ):
            raise ValueError("transport censor coordinate fields are invalid")
        if (
            payload["generation_role"] == "topology"
            and payload["sample_index"] is not None
        ):
            raise ValueError("topology transport censor sample_index must be null")
        if payload["generation_role"] == "self_consistency" and (
            isinstance(payload["sample_index"], bool)
            or not isinstance(payload["sample_index"], int)
            or payload["sample_index"] < 0
        ):
            raise ValueError(
                "self-consistency transport censor sample_index must be non-negative"
            )
    return copy.deepcopy(payload)
