"""The single OpenRouter chat-completion request contract (MASTER_SPEC section 4.2).

Config keys consumed (via the ``sampling`` mapping, normally ``cfg["sampling"]``):

* ``sampling.temperature``      -> body ``temperature`` (required)
* ``sampling.top_p``            -> body ``top_p`` (required)
* ``sampling.api_seed_enabled`` -> must be ``False`` (no ``seed`` is ever sent)
* ``sampling.response_cache``   -> must be ``False`` (``X-OpenRouter-Cache: false`` is always sent)
* ``sampling.n_per_request``    -> must be ``1``
* ``sampling.stream``           -> optional; must be ``False``

Any other key raises ``ValueError``: a knob we do not read must not look configurable (R9).

``provider.allow_fallbacks``/``require_parameters`` are not read from config here: the body hard-codes the only values
``config.validate_config`` accepts (``False``/``True``). Budget comes from the node: ``reasoning.max_tokens`` is
``spec.reasoning_budget`` and ``max_tokens`` is ``spec.max_completion_tokens`` (budget + headroom).
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from agent_allocation.contracts import STUDY_ID, EndpointLock, GenerationSpec

REQUEST_HEADERS: Mapping[str, str] = MappingProxyType({
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-OpenRouter-Cache": "false",
    "X-Title": STUDY_ID,
})

# httpx request-extension key under which ``OpenRouterClient.generate`` hands the node coordinates to the transport.
# Extensions are passed to the transport object only and are never serialized onto the wire; the fake transport uses
# them to look up scripted replies. Nothing in them is sent to OpenRouter.
REQUEST_EXTENSION_KEY = "agent_allocation.request"

# Parameters the locked endpoint must advertise; with ``require_parameters: true`` OpenRouter refuses to route to an
# endpoint lacking one, so we refuse to build the request instead of discovering it with a paid call.
REQUIRED_ENDPOINT_PARAMETERS: tuple[str, ...] = ("max_tokens", "reasoning", "temperature", "top_p")

ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_KNOWN_SAMPLING_KEYS = frozenset({"temperature", "top_p", "api_seed_enabled", "response_cache", "n_per_request",
                                  "stream"})
_TIER_MARKERS = ("/flex", "/priority", "/batch", ":free")


def _number(sampling: Mapping[str, Any], key: str, low: float, high: float) -> float:
    if key not in sampling:
        raise ValueError(f"sampling.{key} is required")
    value = sampling[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"sampling.{key} must be a finite number, got {value!r}")
    if not low <= value <= high:
        raise ValueError(f"sampling.{key}={value!r} is outside [{low}, {high}]")
    return float(value)


def _check_sampling(sampling: Mapping[str, Any]) -> tuple[float, float]:
    unknown = set(sampling) - _KNOWN_SAMPLING_KEYS
    if unknown:
        raise ValueError(f"unknown sampling keys {sorted(unknown)}: not part of the request contract")
    if sampling.get("api_seed_enabled", False) is not False:
        raise ValueError("sampling.api_seed_enabled must be False: no API seed is sent")
    if sampling.get("response_cache", False) is not False:
        raise ValueError("sampling.response_cache must be False: independent draws cannot be cache hits")
    if sampling.get("n_per_request", 1) != 1 or isinstance(sampling.get("n_per_request", 1), bool):
        raise ValueError("sampling.n_per_request must be 1")
    if sampling.get("stream", False) is not False:
        raise ValueError("streaming is disabled: sampling.stream must be False")
    return _number(sampling, "temperature", 0.0, 2.0), _number(sampling, "top_p", 0.0, 1.0)


def check_lock_for_request(lock: EndpointLock) -> None:
    """Raise ``ValueError`` when ``lock`` cannot carry a study request (never pick a tier variant by accident)."""
    tag = lock.provider_tag
    if not tag or tag != tag.strip():
        raise ValueError(f"endpoint lock has no usable provider tag: {tag!r}")
    lowered = tag.lower()
    if any(marker in lowered for marker in _TIER_MARKERS):
        raise ValueError(f"endpoint lock tag {tag!r} is a non-standard tier variant")
    if lock.service_tier != "standard":
        raise ValueError(f"endpoint lock service tier {lock.service_tier!r} is not 'standard'")
    missing = [p for p in REQUIRED_ENDPOINT_PARAMETERS if p not in lock.supported_parameters]
    if missing:
        raise ValueError(f"locked endpoint {tag!r} does not advertise required parameters {missing}")


def spec_lock_problems(spec: GenerationSpec, lock: EndpointLock) -> list[str]:
    """Why ``spec`` must not be dispatched under ``lock`` (empty list = fine). ``generate`` enforces the model and cap
    itself; the lock-hash check belongs to whoever loads manifests against the write-once lock file (live.py)."""
    problems = []
    if spec.model != lock.model:
        problems.append(f"node {spec.node_id} model {spec.model!r} != lock model {lock.model!r}")
    if spec.endpoint_lock_hash != lock.lock_hash:
        problems.append(f"node {spec.node_id} was built for endpoint lock {spec.endpoint_lock_hash[:12]}..., "
                        f"not {lock.lock_hash[:12]}...")
    if spec.max_completion_tokens > lock.max_completion_tokens:
        problems.append(f"node {spec.node_id} max_tokens {spec.max_completion_tokens} > endpoint cap "
                        f"{lock.max_completion_tokens}")
    try:
        check_lock_for_request(lock)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def _check_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("messages must be a non-empty sequence of {role, content} mappings")
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} is not a mapping")
        extra = set(message) - {"role", "content"}
        if extra:
            raise ValueError(f"message {index} has keys outside the contract: {sorted(extra)}")
        role, content = message.get("role"), message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"message {index} has role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"message {index} content must be a string")
        out.append({"role": role, "content": content})
    return out


def build_request_body(spec: GenerationSpec, messages: Sequence[Mapping[str, str]], lock: EndpointLock,
                       sampling: Mapping[str, Any]) -> dict[str, Any]:
    """The exact JSON body for one generation node. No seed, no ``reasoning.effort``, no tools/plugins/transforms."""
    if spec.model != lock.model:
        raise ValueError(f"spec model {spec.model!r} differs from the endpoint lock model {lock.model!r}")
    check_lock_for_request(lock)
    if spec.max_completion_tokens > lock.max_completion_tokens:
        raise ValueError(f"max_tokens {spec.max_completion_tokens} exceeds the locked endpoint cap "
                         f"{lock.max_completion_tokens}")
    temperature, top_p = _check_sampling(sampling)
    return {
        "model": lock.model,
        "messages": _check_messages(messages),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": spec.max_completion_tokens,
        "reasoning": {"max_tokens": spec.reasoning_budget, "exclude": False},
        "provider": {"only": [lock.provider_tag], "allow_fallbacks": False, "require_parameters": True},
        "stream": False,
    }


def request_coordinates(spec: GenerationSpec, attempt_id: str) -> dict[str, Any]:
    """Node coordinates attached as a transport extension (never sent). Used by the fake and by diagnostics."""
    return {
        "node_id": spec.node_id,
        "attempt_id": attempt_id,
        "run_id": spec.run_id,
        "partition_id": spec.partition_id,
        "item_id": spec.item_id,
        "benchmark": spec.benchmark,
        "kind": getattr(spec.kind, "value", spec.kind),
        "root_slot": spec.root_slot,
        "team_size": spec.team_size,
        "update_round": spec.update_round,
        "draw_id": spec.draw_id,
        "repetition_id": spec.repetition_id,
        "reasoning_budget": spec.reasoning_budget,
        "completion_headroom": spec.completion_headroom,
    }
