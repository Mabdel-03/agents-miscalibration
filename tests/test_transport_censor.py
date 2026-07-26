from __future__ import annotations

import copy
import hashlib
import json

import httpx
import pytest
from openai import InternalServerError

from agents_scaling.experiment import transport_censor as protocol


def _request(*, role: str = "topology") -> dict:
    return {
        "qid": "fixture:0",
        "agent_id": "agent0",
        "round": 0,
        "generation_role": role,
        "sample_index": None if role == "topology" else 2,
        "seed": 17,
        "peer_context": {"sha256": "0" * 64, "utf8_bytes": 0},
        "max_tokens": 4096,
        "elicit_cot": True,
    }


def _attempt(request: dict) -> dict:
    return {
        "attempt_id": "a" * 32,
        "coordinate_key": (
            "topology:agent0:0"
            if request["generation_role"] == "topology"
            else "self_consistency:agent0:2"
        ),
        "request_sha256": protocol.canonical_sha256(request),
        "endpoint_generation": "endpoint-g1",
        "started_at": 100.0,
    }


def _censor(
    error: BaseException | None,
    *,
    classification: str = protocol.TRANSPORT_CENSOR_CLASS_PRODUCER_EXCEPTION,
    role: str = "topology",
) -> dict:
    request = _request(role=role)
    return protocol.build_transport_censor(
        request=request,
        attempt=_attempt(request),
        classification=classification,
        censored_at=101.0,
        error=error,
    )


def test_protocol_hash_exactly_covers_error_envelope_and_transport_contract() -> None:
    spec = protocol.transport_censor_protocol_spec()
    envelope = spec["exception_envelope"]
    assert envelope == {
        "utf8_encoding_errors": "replace",
        "readable_message_max_utf8_bytes": (
            protocol.TRANSPORT_CENSOR_ERROR_MESSAGE_MAX_UTF8_BYTES
        ),
        "fully_qualified_exception_type": True,
        "redaction_policy_version": (
            protocol.TRANSPORT_CENSOR_ERROR_REDACTION_POLICY_VERSION
        ),
        "sensitive_message_regex": (
            protocol.TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN
        ),
        "sensitive_message_regex_sha256": hashlib.sha256(
            protocol.TRANSPORT_CENSOR_SENSITIVE_MESSAGE_PATTERN.encode("utf-8")
        ).hexdigest(),
        "redaction_marker": protocol.TRANSPORT_CENSOR_ERROR_REDACTION_MARKER,
        "non_sensitive_digest_state": "full_message",
        "non_sensitive_digest": "sha256(full UTF-8 message bytes)",
        "sensitive_digest_state": "withheld_sensitive",
        "sensitive_digest": None,
        "process_restart": {
            "error_detail_state": "process_restart",
            "all_exception_envelope_fields": None,
        },
    }
    assert spec["single_attempt_transport"] == {
        "openai_sdk_max_retries": 0,
        "chat_completions_create_calls_per_journaled_attempt": 1,
        "stochastic_chat_backoff_decorator": False,
    }
    canonical = json.dumps(
        spec, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == (
        protocol.TRANSPORT_CENSOR_PROTOCOL_HASH
    )


def test_safe_message_retains_fully_qualified_type_and_full_message_digest() -> None:
    message = "upstream closed the socket after accepting the request"
    censor = _censor(RuntimeError(message))
    assert censor["error_type"] == "builtins.RuntimeError"
    assert censor["error_message"] == message
    assert censor["error_message_digest_state"] == "full_message"
    assert censor["error_message_sha256"] == hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()
    assert censor["error_message_utf8_bytes"] == len(message.encode("utf-8"))
    assert censor["error_message_truncated"] is False
    assert censor["error_message_redacted"] is False


@pytest.mark.parametrize(
    "message,secret",
    [
        ("Authorization: Bearer top-secret-value", "top-secret-value"),
        ("Proxy-Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("api_key=sk-abcdefghijklmnop", "sk-abcdefghijklmnop"),
        ("refresh_token=hunter2", "hunter2"),
        ("connect https://alice:password123@example.invalid", "password123"),
    ],
)
def test_secret_bearing_message_withholds_text_and_digest(
    message: str, secret: str
) -> None:
    censor = _censor(RuntimeError(message))
    assert censor["error_message"] == (
        protocol.TRANSPORT_CENSOR_ERROR_REDACTION_MARKER
    )
    assert secret not in json.dumps(censor, sort_keys=True)
    assert censor["error_message_digest_state"] == "withheld_sensitive"
    assert censor["error_message_sha256"] is None
    assert censor["error_message_utf8_bytes"] == len(message.encode("utf-8"))
    assert censor["error_message_redacted"] is True


def test_long_unicode_message_is_utf8_safe_bounded_and_binds_full_message() -> None:
    message = ("x" * 511) + "🙂🙂"
    censor = _censor(RuntimeError(message))
    assert len(censor["error_message"].encode("utf-8")) <= 512
    censor["error_message"].encode("utf-8", errors="strict")
    assert censor["error_message_truncated"] is True
    assert censor["error_message_utf8_bytes"] == len(message.encode("utf-8"))
    assert censor["error_message_sha256"] == hashlib.sha256(
        message.encode("utf-8")
    ).hexdigest()


def test_process_restart_uses_explicit_all_null_error_sentinel() -> None:
    censor = _censor(
        None,
        classification=protocol.TRANSPORT_CENSOR_CLASS_INTERRUPTED,
    )
    assert censor["error_detail_state"] == "process_restart"
    for field in (
        "error_type",
        "error_message",
        "error_message_digest_state",
        "error_message_sha256",
        "error_message_utf8_bytes",
        "error_message_truncated",
        "error_message_redacted",
    ):
        assert censor[field] is None


def test_api_status_error_has_distinct_ambiguous_classification() -> None:
    request = httpx.Request(
        "POST", "http://unused.invalid/v1/chat/completions"
    )
    error = InternalServerError(
        "model server restarted after accepting request",
        response=httpx.Response(500, request=request),
        body=None,
    )
    censor = _censor(
        error,
        classification=protocol.TRANSPORT_CENSOR_CLASS_STATUS,
    )
    assert censor["error_type"] == "openai.InternalServerError"
    assert censor["error_classification"] == (
        protocol.TRANSPORT_CENSOR_CLASS_STATUS
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("error_type", "RuntimeError"),
        ("error_message_digest_state", "full_message"),
        ("error_message_sha256", "f" * 64),
        ("sampling_attempt_count", 2),
        ("request_sha256", "f" * 64),
        ("endpoint_generation", "mixed"),
    ],
)
def test_transport_censor_envelope_tampering_is_rejected(
    field: str, value: object
) -> None:
    request = _request()
    attempt = _attempt(request)
    original = protocol.build_transport_censor(
        request=request,
        attempt=attempt,
        classification=protocol.TRANSPORT_CENSOR_CLASS_PRODUCER_EXCEPTION,
        censored_at=101.0,
        error=RuntimeError("Authorization: Bearer secret-value"),
    )
    tampered = copy.deepcopy(original)
    tampered[field] = value
    with pytest.raises(ValueError):
        protocol.validate_transport_censor(
            tampered, request=request, attempt=attempt
        )
