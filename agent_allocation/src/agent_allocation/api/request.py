"""The single OpenRouter chat-completion request contract (MASTER_SPEC section 4.2).

Config keys consumed (via the ``sampling`` mapping, normally ``cfg["sampling"]``):

* ``sampling.temperature``      -> body ``temperature`` (required)
* ``sampling.top_p``            -> body ``top_p`` (required)
* ``sampling.api_seed_enabled`` -> must be ``False`` (no ``seed`` is ever sent)
* ``sampling.response_cache``   -> must be ``False`` (``X-OpenRouter-Cache: false`` is always sent)
* ``sampling.n_per_request``    -> must be ``1``
* ``sampling.stream``           -> optional; must be ``False``

Any other key raises ``RequestContractError`` (a ``ValueError``): a knob we do not read must not look configurable
(R9). Every validation error here is raised before anything is sent.

Forced mode (``forced.enabled``; built by ``api.forced.ForcedClient``): ``build_forced_think_body`` /
``build_forced_answer_body`` produce raw ``/completions`` bodies (``prompt``, ``logit_bias``, ``usage.include``, no
``reasoning`` field); ``sampling.top_k`` is accepted only there (and then must be advertised by the endpoint); the
lock must advertise ``FORCED_REQUIRED_ENDPOINT_PARAMETERS``. The chat body above is unchanged.

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
# Forced mode (raw ``/completions``, ``forced.enabled``): no ``reasoning`` field is sent; the budget is enforced by
# ``logit_bias`` bans plus ``max_tokens``. ``top_k`` is additionally required when the sampling config sends it.
FORCED_REQUIRED_ENDPOINT_PARAMETERS: tuple[str, ...] = ("logit_bias", "max_tokens", "temperature", "top_p")

class RequestContractError(ValueError):
    """Raised before any byte is sent: the request would break the study contract (spec/lock model or cap, tier
    variant, sampling knobs, message shape, attempt id). The attempt provably never left the process, so the
    dispatcher may release its reservation; it is a configuration/programming blocker, so open the circuit and never
    retry it as a transport failure."""


ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_KNOWN_SAMPLING_KEYS = frozenset({"temperature", "top_p", "api_seed_enabled", "response_cache", "n_per_request",
                                  "stream"})
_FORCED_SAMPLING_KEYS = _KNOWN_SAMPLING_KEYS | {"top_k"}
_TIER_MARKERS = ("/flex", "/priority", "/batch", ":free")


def _number(sampling: Mapping[str, Any], key: str, low: float, high: float) -> float:
    if key not in sampling:
        raise RequestContractError(f"sampling.{key} is required")
    value = sampling[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RequestContractError(f"sampling.{key} must be a finite number, got {value!r}")
    if not low <= value <= high:
        raise RequestContractError(f"sampling.{key}={value!r} is outside [{low}, {high}]")
    return float(value)


def _check_sampling(sampling: Mapping[str, Any], *, forced: bool = False) -> tuple[float, float]:
    """``(temperature, top_p)``; ``top_k`` is a known key only in forced mode (checked by ``forced_top_k``)."""
    unknown = set(sampling) - (_FORCED_SAMPLING_KEYS if forced else _KNOWN_SAMPLING_KEYS)
    if unknown:
        raise RequestContractError(f"unknown sampling keys {sorted(unknown)}: not part of the request contract")
    if sampling.get("api_seed_enabled", False) is not False:
        raise RequestContractError("sampling.api_seed_enabled must be False: no API seed is sent")
    if sampling.get("response_cache", False) is not False:
        raise RequestContractError("sampling.response_cache must be False: independent draws cannot be cache hits")
    if sampling.get("n_per_request", 1) != 1 or isinstance(sampling.get("n_per_request", 1), bool):
        raise RequestContractError("sampling.n_per_request must be 1")
    if sampling.get("stream", False) is not False:
        raise RequestContractError("streaming is disabled: sampling.stream must be False")
    return _number(sampling, "temperature", 0.0, 2.0), _number(sampling, "top_p", 0.0, 1.0)


def forced_top_k(sampling: Mapping[str, Any]) -> int | None:
    """``sampling.top_k`` for a forced request: absent/None -> not sent; otherwise a positive integer."""
    value = sampling.get("top_k")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RequestContractError(f"sampling.top_k must be a positive integer, got {value!r}")
    return value


def check_lock_for_request(lock: EndpointLock, *, required: Sequence[str] = REQUIRED_ENDPOINT_PARAMETERS) -> None:
    """Raise ``RequestContractError`` when ``lock`` cannot carry a study request (never a tier variant by accident)."""
    tag = lock.provider_tag
    if not tag or tag != tag.strip():
        raise RequestContractError(f"endpoint lock has no usable provider tag: {tag!r}")
    lowered = tag.lower()
    if any(marker in lowered for marker in _TIER_MARKERS):
        raise RequestContractError(f"endpoint lock tag {tag!r} is a non-standard tier variant")
    if lock.service_tier != "standard":
        raise RequestContractError(f"endpoint lock service tier {lock.service_tier!r} is not 'standard'")
    missing = [p for p in required if p not in lock.supported_parameters]
    if missing:
        raise RequestContractError(f"locked endpoint {tag!r} does not advertise required parameters {missing}")


def spec_lock_problems(spec: GenerationSpec, lock: EndpointLock, *, forced: bool = False) -> list[str]:
    """Why ``spec`` must not be dispatched under ``lock`` (empty list = fine). ``generate`` enforces the model and cap
    itself; the lock-hash check belongs to whoever loads manifests against the write-once lock file (live.py)."""
    problems = []
    if spec.model != lock.model:
        problems.append(f"node {spec.node_id} model {spec.model!r} != lock model {lock.model!r}")
    if spec.endpoint_lock_hash != lock.lock_hash:
        problems.append(f"node {spec.node_id} was built for endpoint lock {spec.endpoint_lock_hash[:12]}..., "
                        f"not {lock.lock_hash[:12]}...")
    if forced:
        # two calls: THINK max_tokens = budget, ANSWER max_tokens = headroom; each must fit the endpoint cap
        if max(spec.reasoning_budget, spec.completion_headroom) > lock.max_completion_tokens:
            problems.append(f"node {spec.node_id} forced call max_tokens "
                            f"{max(spec.reasoning_budget, spec.completion_headroom)} > endpoint cap "
                            f"{lock.max_completion_tokens}")
    elif spec.max_completion_tokens > lock.max_completion_tokens:
        problems.append(f"node {spec.node_id} max_tokens {spec.max_completion_tokens} > endpoint cap "
                        f"{lock.max_completion_tokens}")
    try:
        check_lock_for_request(lock, required=FORCED_REQUIRED_ENDPOINT_PARAMETERS if forced
                               else REQUIRED_ENDPOINT_PARAMETERS)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def _check_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not messages:
        raise RequestContractError("messages must be a non-empty sequence of {role, content} mappings")
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise RequestContractError(f"message {index} is not a mapping")
        extra = set(message) - {"role", "content"}
        if extra:
            raise RequestContractError(f"message {index} has keys outside the contract: {sorted(extra)}")
        role, content = message.get("role"), message.get("content")
        if role not in ALLOWED_ROLES:
            raise RequestContractError(f"message {index} has role {role!r}")
        if not isinstance(content, str):
            raise RequestContractError(f"message {index} content must be a string")
        out.append({"role": role, "content": content})
    return out


def build_request_body(spec: GenerationSpec, messages: Sequence[Mapping[str, str]], lock: EndpointLock,
                       sampling: Mapping[str, Any]) -> dict[str, Any]:
    """The exact JSON body for one generation node. No seed, no ``reasoning.effort``, no tools/plugins/transforms."""
    if spec.model != lock.model:
        raise RequestContractError(f"spec model {spec.model!r} differs from the endpoint lock model {lock.model!r}")
    check_lock_for_request(lock)
    if spec.max_completion_tokens > lock.max_completion_tokens:
        raise RequestContractError(f"max_tokens {spec.max_completion_tokens} exceeds the locked endpoint cap "
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


def _forced_common(spec: GenerationSpec, lock: EndpointLock, sampling: Mapping[str, Any],
                   prompt: str) -> dict[str, Any]:
    if spec.model != lock.model:
        raise RequestContractError(f"spec model {spec.model!r} differs from the endpoint lock model {lock.model!r}")
    temperature, top_p = _check_sampling(sampling, forced=True)
    top_k = forced_top_k(sampling)
    required = FORCED_REQUIRED_ENDPOINT_PARAMETERS + (("top_k",) if top_k is not None else ())
    check_lock_for_request(lock, required=required)
    if not isinstance(prompt, str) or not prompt:
        raise RequestContractError("forced prompt must be a non-empty string")
    body: dict[str, Any] = {"model": lock.model, "prompt": prompt, "temperature": temperature, "top_p": top_p}
    if top_k is not None:
        body["top_k"] = top_k
    return body


def _forced_tail(lock: EndpointLock) -> dict[str, Any]:
    return {"provider": {"only": [lock.provider_tag], "allow_fallbacks": False, "require_parameters": True},
            "usage": {"include": True}, "stream": False}


def build_forced_think_body(spec: GenerationSpec, prompt: str, lock: EndpointLock, sampling: Mapping[str, Any], *,
                            ban_token_ids: Sequence[int], ban_bias: int) -> dict[str, Any]:
    """THINK call of a forced node: the rendered chat prompt ending ``<think>\n``, ``max_tokens`` = the node's
    reasoning budget, ``logit_bias`` = ``ban_bias`` on every banned id (``</think>``, ``<|im_end|>``,
    ``<|endoftext|>``) so the call can only end at the budget. No ``reasoning`` field, no seed."""
    if spec.reasoning_budget > lock.max_completion_tokens:
        raise RequestContractError(f"THINK max_tokens {spec.reasoning_budget} exceeds the locked endpoint cap "
                                   f"{lock.max_completion_tokens}")
    ids = list(ban_token_ids)
    if not ids or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in ids):
        raise RequestContractError(f"forced.ban_token_ids must be non-negative integers, got {ids!r}")
    if isinstance(ban_bias, bool) or not isinstance(ban_bias, (int, float)) or not -100 <= ban_bias < 0:
        raise RequestContractError(f"forced.ban_bias must be in [-100, 0), got {ban_bias!r}")
    body = _forced_common(spec, lock, sampling, prompt)
    body["max_tokens"] = spec.reasoning_budget
    body["logit_bias"] = {str(t): ban_bias for t in ids}
    body.update(_forced_tail(lock))
    return body


def build_natural_body(spec: GenerationSpec, prompt: str, lock: EndpointLock,
                       sampling: Mapping[str, Any]) -> dict[str, Any]:
    """A NATURAL (unforced) reference call on the same raw prompt: no ``logit_bias``, the model closes its own
    reasoning and answers in the same call; ``max_tokens`` = reasoning budget + completion headroom."""
    cap = spec.reasoning_budget + spec.completion_headroom
    if cap > lock.max_completion_tokens:
        raise RequestContractError(f"NATURAL max_tokens {cap} exceeds the locked endpoint cap "
                                   f"{lock.max_completion_tokens}")
    body = _forced_common(spec, lock, sampling, prompt)
    body["max_tokens"] = cap
    body.update(_forced_tail(lock))
    return body


def build_forced_answer_body(spec: GenerationSpec, prompt: str, lock: EndpointLock,
                             sampling: Mapping[str, Any]) -> dict[str, Any]:
    """ANSWER call of a forced node: THINK prompt + the realized reasoning + the close text, ``max_tokens`` = the
    node's completion headroom, no ``logit_bias``."""
    if spec.completion_headroom > lock.max_completion_tokens:
        raise RequestContractError(f"ANSWER max_tokens {spec.completion_headroom} exceeds the locked endpoint cap "
                                   f"{lock.max_completion_tokens}")
    body = _forced_common(spec, lock, sampling, prompt)
    body["max_tokens"] = spec.completion_headroom
    body.update(_forced_tail(lock))
    return body


def build_forced_probe_body(lock: EndpointLock, sampling: Mapping[str, Any], prompt: str, *, max_tokens: int,
                            logit_bias: Mapping[str, int | float] | None) -> dict[str, Any]:
    """A Gate 0 probe body (banana / ``</think>``-id tests): same contract, explicit ``max_tokens`` and bias."""
    temperature, top_p = _check_sampling(sampling, forced=True)
    top_k = forced_top_k(sampling)
    check_lock_for_request(lock, required=FORCED_REQUIRED_ENDPOINT_PARAMETERS + (("top_k",) if top_k else ()))
    body: dict[str, Any] = {"model": lock.model, "prompt": prompt, "temperature": temperature, "top_p": top_p}
    if top_k is not None:
        body["top_k"] = top_k
    body["max_tokens"] = int(max_tokens)
    if logit_bias:
        body["logit_bias"] = {str(k): v for k, v in logit_bias.items()}
    body.update(_forced_tail(lock))
    return body


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


# ---------------------------------------------------------------------------------------------------------------------
# dynorch_v1 (WP-A, contract §5): the capped -- not forced -- raw completion body
# ---------------------------------------------------------------------------------------------------------------------

CAPPED_REQUIRED_ENDPOINT_PARAMETERS: tuple[str, ...] = ("max_tokens", "temperature", "top_p")


def build_capped_body(prompt: str, lock: EndpointLock, sampling: Mapping[str, Any], *,
                      max_tokens: int) -> dict[str, Any]:
    """dynorch (contract §5, plan B2): one raw ``POST /completions`` on the rendered prompt with ``max_tokens`` =
    the call's grant and NO ``logit_bias`` -- the call is capped, never forced, so it may stop naturally below its
    grant. Keys are exactly ``_forced_common`` minus ``logit_bias`` plus ``_forced_tail`` (``model, prompt,
    temperature, top_p[, top_k], max_tokens, provider{only, allow_fallbacks:false, require_parameters:true},
    usage{include:true}, stream:false``), i.e. inside ``api/fake.py::completions_contract_violations``. The lock must
    advertise ``max_tokens, temperature, top_p`` (+ ``top_k`` when the sampling block sends it; plan §4)."""
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise RequestContractError(f"capped max_tokens must be a positive integer, got {max_tokens!r}")
    if max_tokens > lock.max_completion_tokens:
        raise RequestContractError(f"capped max_tokens {max_tokens} exceeds the locked endpoint cap "
                                   f"{lock.max_completion_tokens}")
    temperature, top_p = _check_sampling(sampling, forced=True)
    top_k = forced_top_k(sampling)
    required = CAPPED_REQUIRED_ENDPOINT_PARAMETERS + (("top_k",) if top_k is not None else ())
    check_lock_for_request(lock, required=required)
    if not isinstance(prompt, str) or not prompt:
        raise RequestContractError("capped prompt must be a non-empty string")
    body: dict[str, Any] = {"model": lock.model, "prompt": prompt, "temperature": temperature, "top_p": top_p}
    if top_k is not None:
        body["top_k"] = top_k
    body["max_tokens"] = max_tokens
    body.update(_forced_tail(lock))
    assert "logit_bias" not in body
    return body
