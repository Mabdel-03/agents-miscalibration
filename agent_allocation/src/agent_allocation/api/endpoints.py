"""Exact standard-tier endpoint selection from a saved ``GET /models/{model}/endpoints`` snapshot (spec 4.1).

``select_standard_endpoint`` never invents or rewrites a tag: it locks the endpoint whose ``tag`` equals
``preferred_tag`` character for character (``google-ai-studio`` does not match ``google-ai-studio/flex``), and raises
``EndpointBlocked`` when that endpoint is absent, duplicated, a tier variant (``/flex``, ``/priority``, ``/batch``,
``:free``), for a preview/experimental/variant model, lacking a required parameter, or missing pricing/limits.
Prices are copied as the snapshot's decimal strings. Volatile listing fields (status, uptime, latency) are reported by
``endpoint_candidates`` but never enter the lock.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from agent_allocation.api.request import REQUIRED_ENDPOINT_PARAMETERS
from agent_allocation.contracts import EndpointLock

TIER_MARKERS: tuple[str, ...] = ("/flex", "/priority", "/batch", ":free")
_UNSTABLE_MODEL = re.compile(r"(preview|experimental|(^|[-_/.])exp($|[-_/.\d]))", re.IGNORECASE)


class EndpointBlocked(RuntimeError):
    pass


def model_problems(model: str) -> list[str]:
    problems = []
    if not isinstance(model, str) or "/" not in model:
        return [f"{model!r} is not an OpenRouter model slug"]
    if ":" in model:
        problems.append(f"model {model!r} is a routing/pricing variant (':' suffix such as :free)")
    if model.startswith("openrouter/"):
        problems.append(f"model {model!r} is automatic routing")
    if _UNSTABLE_MODEL.search(model):
        problems.append(f"model {model!r} is a preview/experimental release")
    return problems


def tag_problems(tag: Any) -> list[str]:
    if not isinstance(tag, str) or not tag or tag != tag.strip():
        return [f"endpoint tag {tag!r} is not a usable tag"]
    lowered = tag.lower()
    return [f"tag {tag!r} is a non-standard tier variant ({marker})" for marker in TIER_MARKERS if marker in lowered]


def _price(pricing: Mapping[str, Any], key: str, tag: str) -> str:
    value = pricing.get(key)
    if isinstance(value, bool) or value is None:
        raise EndpointBlocked(f"endpoint {tag!r} has no pricing.{key}")
    text = str(value).strip()
    try:
        amount = Decimal(text)
    except InvalidOperation:
        raise EndpointBlocked(f"endpoint {tag!r} pricing.{key}={value!r} is not a decimal") from None
    if not amount.is_finite() or amount <= 0:
        raise EndpointBlocked(f"endpoint {tag!r} pricing.{key}={value!r} is not a positive price "
                              f"(free/zero-priced endpoints are not standard)")
    return text


def _positive_int(endpoint: Mapping[str, Any], key: str, tag: str) -> int:
    value = endpoint.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EndpointBlocked(f"endpoint {tag!r} has no positive integer {key} (got {value!r})")
    return value


def _endpoints(snapshot: Mapping[str, Any], model: str) -> list[Mapping[str, Any]]:
    data = snapshot.get("data") if isinstance(snapshot, Mapping) else None
    if not isinstance(data, Mapping):
        raise EndpointBlocked("endpoints snapshot has no 'data' object")
    if data.get("id") != model:
        raise EndpointBlocked(f"endpoints snapshot is for {data.get('id')!r}, not {model!r}")
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        raise EndpointBlocked("endpoints snapshot has no endpoint list")
    return [e for e in endpoints if isinstance(e, Mapping)]


def endpoint_candidates(snapshot: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    """Capability-report rows: every listed endpoint with its tag, prices, status and why it is (not) selectable."""
    rows = []
    for endpoint in _endpoints(snapshot, model):
        tag = endpoint.get("tag")
        problems = model_problems(model) + tag_problems(tag)
        if endpoint.get("model_id") not in (None, model):
            problems.append(f"model_id {endpoint.get('model_id')!r} != {model!r}")
        supported = endpoint.get("supported_parameters") or []
        missing = [p for p in REQUIRED_ENDPOINT_PARAMETERS if p not in supported]
        if missing:
            problems.append(f"missing required parameters {missing}")
        pricing = endpoint.get("pricing") if isinstance(endpoint.get("pricing"), Mapping) else {}
        rows.append({
            "tag": tag,
            "provider_name": endpoint.get("provider_name"),
            "prompt_price": pricing.get("prompt"),
            "completion_price": pricing.get("completion"),
            "internal_reasoning_price": pricing.get("internal_reasoning"),
            "discount": pricing.get("discount"),
            "status": endpoint.get("status"),
            "uptime_last_30m": endpoint.get("uptime_last_30m"),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "context_length": endpoint.get("context_length"),
            "standard_selectable": not problems,
            "problems": problems,
        })
    return rows


def select_standard_endpoint(snapshot: Mapping[str, Any], model: str, *, preferred_tag: str = "google-ai-studio",
                             snapshot_path: str = "", snapshot_sha256: str = "", fetched_at: str = "") -> EndpointLock:
    problems = model_problems(model) + tag_problems(preferred_tag)
    if problems:
        raise EndpointBlocked("; ".join(problems))
    matches = [e for e in _endpoints(snapshot, model) if e.get("tag") == preferred_tag]
    if not matches:
        listed = sorted(str(e.get("tag")) for e in _endpoints(snapshot, model))
        raise EndpointBlocked(f"no endpoint tagged exactly {preferred_tag!r} for {model}; listed tags: {listed}")
    if len(matches) > 1:
        raise EndpointBlocked(f"{len(matches)} endpoints share the tag {preferred_tag!r}; refusing to guess")
    endpoint = matches[0]
    if endpoint.get("model_id") not in (None, model):
        raise EndpointBlocked(f"endpoint {preferred_tag!r} serves {endpoint.get('model_id')!r}, not {model!r}")
    provider_name = endpoint.get("provider_name")
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise EndpointBlocked(f"endpoint {preferred_tag!r} has no provider_name")
    supported = endpoint.get("supported_parameters")
    if not isinstance(supported, list) or not all(isinstance(p, str) for p in supported):
        raise EndpointBlocked(f"endpoint {preferred_tag!r} has no supported_parameters list")
    missing = [p for p in REQUIRED_ENDPOINT_PARAMETERS if p not in supported]
    if missing:
        raise EndpointBlocked(f"endpoint {preferred_tag!r} does not support required parameters {missing}")
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        raise EndpointBlocked(f"endpoint {preferred_tag!r} has no pricing")
    return EndpointLock(
        model=model,
        provider_tag=preferred_tag,
        provider_name=provider_name,
        service_tier="standard",
        input_price_per_token=_price(pricing, "prompt", preferred_tag),
        output_price_per_token=_price(pricing, "completion", preferred_tag),
        internal_reasoning_price_per_token=_price(pricing, "internal_reasoning", preferred_tag),
        context_length=_positive_int(endpoint, "context_length", preferred_tag),
        max_completion_tokens=_positive_int(endpoint, "max_completion_tokens", preferred_tag),
        supported_parameters=tuple(sorted(set(supported))),
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha256,
        fetched_at=fetched_at,
    )
