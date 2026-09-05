"""Canonical JSON, semantic seeds and request identity for the study (WP0).

Spec: §3.5 (RFC 8785 canonical JSON, preserve UTF-8), §3.6 (semantic seed, request_id,
alias-by-exact-equality), §6.7 (budget label is *not* part of the identity), §4.1/§4.6
(blind display order by immutable hashes).  Architecture: docs/study_v4/01_architecture.md
§1.2; corrections: 04_critic_corrections.md §4 item 9 (study_seed provenance).

Design rules
* ``jcs`` implements the RFC 8785 subset actually used by identity inputs: ``str``,
  ``int``, ``bool``, ``None``, lists/tuples and string-keyed mappings.  Floats raise —
  decoding floats are serialized as fixed strings by :func:`float_str` inside
  :func:`decoding_hash` so that no platform float formatting can enter an identity.
* Every function here is pure and importable without the rest of the package;
  ``types.py`` imports this module (never the reverse at import time).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from agents_scaling.study.types import Checkpoint, Decoding, SeedKey

# Frozen engine facts that enter ``engine_digest`` (§3.6, §6.7, amendment P4).
VLLM_VERSION = "0.21.0"
ENGINE_DTYPE = "bf16"
REASONING_PARSER = "qwen3"
HOOK_HASH_NONE = "none"
SEMANTIC_SEED_BYTES = 16
ENGINE_SEED_MASK = (1 << 63) - 1


# --------------------------------------------------------------------------- JCS


def _canonical(obj: Any) -> Any:
    """Return ``obj`` converted to a JSON-ready structure in RFC 8785 key order."""
    if isinstance(obj, Enum):
        obj = obj.value
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        raise TypeError(
            "floats are not permitted in identity inputs (RFC 8785 number formatting is "
            "not implemented); serialize them as fixed strings via float_str()"
        )
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_canonical(item) for item in obj]
    if isinstance(obj, Mapping):
        pairs: list[tuple[str, Any]] = []
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"JCS object keys must be str, got {type(key).__name__}")
            pairs.append((key, value))
        # RFC 8785 §3.2.3: sort by UTF-16 code units, not Unicode code points.
        pairs.sort(key=lambda kv: kv[0].encode("utf-16-be"))
        return {key: _canonical(value) for key, value in pairs}
    raise TypeError(f"type {type(obj).__name__} is not JCS-serializable")


def jcs(obj: Any) -> bytes:
    """RFC 8785 canonical JSON (subset): sorted keys, no whitespace, UTF-8, no floats.

    ``ensure_ascii=False`` preserves string content byte-for-byte (§3.5 "do not silently
    normalize code/task strings"); Python's encoder escapes exactly the characters RFC
    8785 requires (``"``, ``\\`` and U+0000–U+001F, lowercase ``\\u00xx``).
    """
    return json.dumps(
        _canonical(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def float_str(value: float) -> str:
    """Fixed textual form of a finite float for hashing (shortest round-trip repr)."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite float in identity input")
    return repr(number)


def sha256_hex(data: bytes | str) -> str:
    """Hex SHA-256 of bytes (str is UTF-8 encoded first)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- seeds


def seed_key_array(key: SeedKey | Sequence[Any]) -> list[Any]:
    """The eight-element JCS array in the frozen §3.6 order."""
    if hasattr(key, "as_array"):
        array = list(key.as_array())
    else:
        array = list(key)
    if len(array) != 8:
        raise ValueError("seed key must have exactly eight components (§3.6)")
    return array


def semantic_seed(study_seed: bytes, key: SeedKey | Sequence[Any]) -> bytes:
    """First 128 bits of HMAC-SHA256(study_seed, JCS([...seed key...])) (§3.6)."""
    if not isinstance(study_seed, (bytes, bytearray)) or len(study_seed) == 0:
        raise ValueError("study_seed must be non-empty bytes")
    digest = hmac.new(bytes(study_seed), jcs(seed_key_array(key)), hashlib.sha256).digest()
    return digest[:SEMANTIC_SEED_BYTES]


def engine_seed(seed16: bytes) -> int:
    """Map a semantic seed to vLLM's signed 64-bit ``seed`` range (§3.6).

    The documented collision check lives with the consumers: ``cells.py`` asserts
    engine-seed uniqueness within every episode and ``store.publish`` asserts that an
    existing record with the same engine seed carries the same request_id.
    """
    if len(seed16) < 8:
        raise ValueError("semantic seed must be at least 8 bytes")
    return int.from_bytes(bytes(seed16[:8]), "big") & ENGINE_SEED_MASK


def blind_order_key(study_seed: bytes, namespace: str, *parts: Any) -> bytes:
    """HMAC for blind display/permutation order of peers, candidates and roots (§4.1, §4.6).

    Sorting by these bytes is independent of confidence, validity and correctness.
    """
    return hmac.new(bytes(study_seed), jcs([namespace, *parts]), hashlib.sha256).digest()


# --------------------------------------------------------------------------- request identity


def input_hash(messages: Sequence[Mapping[str, Any]], chat_template_kwargs: Mapping[str, Any]) -> str:
    """SHA-256 of JCS({messages, chat_template_kwargs}) — the prompt bytes identity (§3.6)."""
    return sha256_hex(
        jcs({"messages": list(messages), "chat_template_kwargs": dict(chat_template_kwargs)})
    )


def decoding_hash(decoding: Decoding | Mapping[str, Any]) -> str:
    """SHA-256 of JCS(decoding with floats as fixed strings) (§3.6; amendment E2 flag)."""
    payload = decoding.as_strings() if hasattr(decoding, "as_strings") else dict(decoding)
    return sha256_hex(jcs(payload))


def engine_digest(checkpoint: Checkpoint) -> str:
    """Frozen engine identity string (§3.6, §6.7; amendment P4)."""
    return (
        f"vllm=={VLLM_VERSION};dtype={ENGINE_DTYPE};tp={int(checkpoint.tp_size)};"
        f"reasoning_parser={REASONING_PARSER};profile={checkpoint.profile}"
    )


def request_id(
    study_id: str,
    model_revision: str,
    tokenizer_revision: str,
    engine_digest: str,
    input_hash: str,
    decoding_hash: str,
    semantic_seed_hex: str,
    local_caps: Mapping[str, Any],
    hook_hash: str = HOOK_HASH_NONE,
) -> str:
    """``SHA256(JCS([study_id, model_revision, tokenizer_revision, engine_digest,
    input_hash, decoding_hash, semantic_seed, local_caps, hook_hash]))`` (§3.6).

    The outer budget label is deliberately absent (§6.7): it never changes a prompt,
    a local cap or the request path, so B-cells alias by construction.
    """
    return sha256_hex(
        jcs(
            [
                study_id,
                model_revision,
                tokenizer_revision,
                engine_digest,
                input_hash,
                decoding_hash,
                semantic_seed_hex,
                dict(local_caps),
                hook_hash,
            ]
        )
    )


def content_sha256(record: Mapping[str, Any], *, exclude: str = "content_sha256") -> str:
    """Integrity hash of an on-disk record without its own hash field (architecture §2.1).

    Records carry floats (timings, FLOPs); they are hashed through the standard JSON
    encoder with sorted keys rather than :func:`jcs`, which forbids floats.  This hash is
    an integrity check on stored bytes, never an identity input.
    """
    body = {k: v for k, v in record.items() if k != exclude}
    return sha256_hex(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    )
