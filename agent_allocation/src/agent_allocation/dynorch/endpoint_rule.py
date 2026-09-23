"""WP-D: the pre-registered provider pinning rule (plan §4; contract §8).

Gate 1 evaluates each forced_v2 shard endpoint (``mancer/fp8``, ``parasail/fp8``) from one
``GET /models/qwen/qwen3.8-27b/endpoints`` snapshot: pass = ``status == 0`` and ``uptime_last_30m >= 95`` and
``supported_parameters ⊇ {max_tokens, temperature, top_p, top_k}`` (thresholds from ``config.dynorch.provider_rule``).
Items keep their forced_v2 shard (``shards/assignment.json``) when it passes; if one shard fails, every item goes to
the other; if both fail, the decision is ``stop`` (stop and notify). The decision is frozen into
``<run_root>/locks/openrouter_endpoints.json`` (one ``EndpointLock`` per shard, minted by
``api/endpoints.py::select_standard_endpoint``) and ``<run_root>/locks/provider_map.json`` (one row per item) and is
never re-evaluated mid-run. Everything here takes the parsed snapshot; the online fetch lives in the gate verb
(``api/client.py::OpenRouterClient.list_endpoints``).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.api.endpoints import select_standard_endpoint
from agent_allocation.contracts import EndpointLock, dump_record, load_record
from agent_allocation.storage._atomic import replace_atomic, write_once

LOCK_LAYOUT = "dynorch/v1"
KEEP, SWITCH, STOP = "keep", "switch", "stop"


class EndpointRuleError(RuntimeError):
    """A snapshot or an input shape the rule cannot evaluate (never a silent default)."""


@dataclass(frozen=True)
class EndpointVerdict:
    """One shard endpoint against the rule (plan §4)."""
    shard_id: str
    tag: str
    listed: bool
    status: int | None
    uptime_last_30m: float | None
    missing_parameters: tuple[str, ...]
    passed: bool
    problems: tuple[str, ...]


@dataclass(frozen=True)
class ProviderDecision:
    """``action`` ∈ {keep, switch, stop}; ``shard_to_tag`` maps every base shard id to the tag it will use."""
    action: str
    shard_to_tag: dict[str, str]
    verdicts: tuple[EndpointVerdict, ...]
    reason: str


def _rule(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    rule = cfg.get("dynorch", {}).get("provider_rule")
    if not isinstance(rule, Mapping):
        raise EndpointRuleError("config.dynorch.provider_rule is missing")
    for key in ("status", "uptime_last_30m_min", "required_parameters"):
        if key not in rule:
            raise EndpointRuleError(f"config.dynorch.provider_rule.{key} is missing")
    return rule


def shard_tags(cfg: Mapping[str, Any]) -> dict[str, str]:
    """``{shard_id: endpoint_tag}`` from ``config.shards`` (the base study's two provider shards)."""
    shards = cfg.get("shards")
    if not isinstance(shards, list) or not shards:
        raise EndpointRuleError("config.shards is missing or empty")
    out: dict[str, str] = {}
    for shard in shards:
        if not isinstance(shard, Mapping) or not shard.get("id") or not shard.get("endpoint_tag"):
            raise EndpointRuleError(f"config.shards entry without id/endpoint_tag: {shard!r}")
        out[str(shard["id"])] = str(shard["endpoint_tag"])
    return out


def _listed_endpoints(snapshot: Mapping[str, Any], model: str) -> list[Mapping[str, Any]]:
    data = snapshot.get("data") if isinstance(snapshot, Mapping) else None
    if not isinstance(data, Mapping) or data.get("id") != model or not isinstance(data.get("endpoints"), list):
        raise EndpointRuleError(f"endpoints snapshot is not a GET /models/{model}/endpoints payload")
    return [e for e in data["endpoints"] if isinstance(e, Mapping)]


def evaluate_endpoint(snapshot: Mapping[str, Any], model: str, *, shard_id: str, tag: str,
                      rule: Mapping[str, Any]) -> EndpointVerdict:
    """Plan §4 pass/fail for one shard's endpoint tag from the parsed snapshot."""
    matches = [e for e in _listed_endpoints(snapshot, model) if e.get("tag") == tag]
    if not matches:
        return EndpointVerdict(shard_id, tag, False, None, None, tuple(rule["required_parameters"]), False,
                               (f"endpoint {tag!r} is not listed for {model}",))
    if len(matches) > 1:
        raise EndpointRuleError(f"{len(matches)} endpoints share the tag {tag!r}; refusing to guess")
    endpoint = matches[0]
    problems: list[str] = []
    status = endpoint.get("status")
    if isinstance(status, bool) or not isinstance(status, int):
        problems.append(f"status {status!r} is not an integer")
        status_value: int | None = None
    else:
        status_value = status
        if status != int(rule["status"]):
            problems.append(f"status {status} != {int(rule['status'])}")
    uptime = endpoint.get("uptime_last_30m")
    if isinstance(uptime, bool) or not isinstance(uptime, (int, float)):
        problems.append(f"uptime_last_30m {uptime!r} is not a number")
        uptime_value: float | None = None
    else:
        uptime_value = float(uptime)
        if uptime_value < float(rule["uptime_last_30m_min"]):
            problems.append(f"uptime_last_30m {uptime_value} < {float(rule['uptime_last_30m_min'])}")
    supported = endpoint.get("supported_parameters")
    if not isinstance(supported, list):
        problems.append("supported_parameters is not a list")
        missing = tuple(rule["required_parameters"])
    else:
        missing = tuple(p for p in rule["required_parameters"] if p not in supported)
        if missing:
            problems.append(f"missing required parameters {list(missing)}")
    return EndpointVerdict(shard_id, tag, True, status_value, uptime_value, missing, not problems, tuple(problems))


def decide(snapshot: Mapping[str, Any], cfg: Mapping[str, Any]) -> ProviderDecision:
    """The frozen rule over every configured shard: keep (all pass), switch (exactly one passing tag takes every
    shard), stop (none pass)."""
    model = str(cfg["primary_model"])
    rule = _rule(cfg)
    tags = shard_tags(cfg)
    verdicts = tuple(evaluate_endpoint(snapshot, model, shard_id=sid, tag=tag, rule=rule) for sid, tag in tags.items())
    passing = [v for v in verdicts if v.passed]
    if len(passing) == len(verdicts):
        return ProviderDecision(KEEP, dict(tags), verdicts, "every shard endpoint passes; items keep their forced_v2 shard")
    if not passing:
        return ProviderDecision(STOP, {}, verdicts, "no shard endpoint passes the rule: stop and notify")
    if len(passing) > 1:
        raise EndpointRuleError("more than one passing endpoint with a failing one: the rule is defined for two shards")
    target = passing[0]
    failed = [v.shard_id for v in verdicts if not v.passed]
    return ProviderDecision(SWITCH, {sid: target.tag for sid in tags},
                            verdicts, f"shard(s) {failed} fail the rule; every item goes to {target.tag!r}")


def provider_map(decision: ProviderDecision, assignment: Mapping[str, str], tags: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Per-item ``{item_id: {forced_v2_shard, shard_id, endpoint_tag}}``. ``shard_id`` is the shard whose lock the
    proxy uses (the passing shard after a switch); ``forced_v2_shard`` is the sealed base assignment (a covariate)."""
    if decision.action == STOP:
        raise EndpointRuleError("no provider map for a stop decision")
    tag_to_shard = {tag: sid for sid, tag in tags.items()}
    rows: dict[str, dict[str, str]] = {}
    for item_id, base_shard in sorted(assignment.items()):
        if base_shard not in decision.shard_to_tag:
            raise EndpointRuleError(f"assignment shard {base_shard!r} for {item_id} is not a configured shard")
        tag = decision.shard_to_tag[base_shard]
        rows[item_id] = {"forced_v2_shard": base_shard, "shard_id": tag_to_shard[tag], "endpoint_tag": tag}
    return rows


def load_assignment(base_run_root: Path) -> dict[str, str]:
    """``shards/assignment.json`` of the sealed base run (read-only)."""
    path = Path(base_run_root) / "shards" / "assignment.json"
    data = json.loads(path.read_text())
    assignment = data.get("assignment") if isinstance(data, Mapping) else None
    if not isinstance(assignment, Mapping) or not assignment:
        raise EndpointRuleError(f"{path} has no 'assignment' object")
    return {str(k): str(v) for k, v in assignment.items()}


def lock_payload(decision: ProviderDecision, snapshot: Mapping[str, Any], cfg: Mapping[str, Any], *,
                 snapshot_path: str = "", snapshot_sha256: str = "", fetched_at: str = "") -> dict[str, Any]:
    """The ``locks/openrouter_endpoints.json`` document: one ``EndpointLock`` per shard id that any item uses,
    minted by ``select_standard_endpoint`` (forced parameter contract, like the base study's shard locks)."""
    if decision.action == STOP:
        raise EndpointRuleError("no lock for a stop decision")
    model = str(cfg["primary_model"])
    tags = shard_tags(cfg)
    used_tags = sorted(set(decision.shard_to_tag.values()))
    locks: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for tag in used_tags:
        shard_id = next(sid for sid, t in tags.items() if t == tag)
        lock = select_standard_endpoint(snapshot, model, preferred_tag=tag, forced=True, snapshot_path=snapshot_path,
                                        snapshot_sha256=snapshot_sha256, fetched_at=fetched_at)
        locks[shard_id] = dump_record(lock)
        hashes[shard_id] = lock.lock_hash
    return {
        "layout": LOCK_LAYOUT,
        "model": model,
        "decision": {"action": decision.action, "reason": decision.reason, "shard_to_tag": dict(decision.shard_to_tag),
                     "verdicts": [asdict(v) for v in decision.verdicts]},
        "locks": locks,                      # keyed by shard id (the Upstream lock map of contract §5)
        "lock_hashes": hashes,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def read_locks(run_root: Path) -> dict[str, EndpointLock]:
    """``{shard_id: EndpointLock}`` from the frozen lock file (raises when absent or of another layout)."""
    path = Path(run_root) / "locks" / "openrouter_endpoints.json"
    data = json.loads(path.read_text())
    if data.get("layout") != LOCK_LAYOUT:
        raise EndpointRuleError(f"{path} is not a {LOCK_LAYOUT} lock file")
    return {sid: load_record(EndpointLock, row) for sid, row in data["locks"].items()}


def read_decision(run_root: Path) -> dict[str, Any]:
    path = Path(run_root) / "locks" / "openrouter_endpoints.json"
    data = json.loads(path.read_text())
    if data.get("layout") != LOCK_LAYOUT or not isinstance(data.get("decision"), Mapping):
        raise EndpointRuleError(f"{path} is not a {LOCK_LAYOUT} lock file")
    return dict(data["decision"])


def freeze(run_root: Path, snapshot: Mapping[str, Any], raw_snapshot: bytes, cfg: Mapping[str, Any], *,
           assignment: Mapping[str, str], fetched_at: str) -> ProviderDecision:
    """Apply the rule and write the two lock files write-once (an existing, different lock refuses; an identical
    rewrite is a no-op). Returns the decision; a ``stop`` decision writes only ``locks/provider_decision.json``."""
    root = Path(run_root)
    locks_dir = root / "locks"
    snapshots = locks_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    digest = sha256(raw_snapshot).hexdigest()
    snap_path = snapshots / f"endpoints_{str(cfg['primary_model']).replace('/', '_')}_{fetched_at}.json"
    if not snap_path.exists():
        replace_atomic(snap_path, raw_snapshot)
    decision = decide(snapshot, cfg)
    decision_doc = {"action": decision.action, "reason": decision.reason, "shard_to_tag": decision.shard_to_tag,
                    "verdicts": [asdict(v) for v in decision.verdicts], "snapshot_sha256": digest,
                    "fetched_at": fetched_at}
    replace_atomic(locks_dir / "provider_decision.json", json.dumps(decision_doc, indent=1, sort_keys=True).encode())
    if decision.action == STOP:
        return decision
    payload = lock_payload(decision, snapshot, cfg, snapshot_path=str(snap_path), snapshot_sha256=digest,
                           fetched_at=fetched_at)
    lock_path = locks_dir / "openrouter_endpoints.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        if existing.get("lock_hashes") != payload["lock_hashes"] or \
                existing.get("decision", {}).get("shard_to_tag") != payload["decision"]["shard_to_tag"]:
            raise EndpointRuleError(f"{lock_path} is frozen with a different decision; the rule is not re-evaluated mid-run")
    else:
        write_once(lock_path, json.dumps(payload, indent=1, sort_keys=True).encode())
    rows = provider_map(decision, assignment, shard_tags(cfg))
    map_path = locks_dir / "provider_map.json"
    map_doc = json.dumps({"layout": LOCK_LAYOUT, "items": rows, "decision": decision.action}, indent=1,
                         sort_keys=True).encode()
    if map_path.exists():
        if json.loads(map_path.read_text()).get("items") != rows:
            raise EndpointRuleError(f"{map_path} is frozen with a different item map")
    else:
        write_once(map_path, map_doc)
    return decision


def counts_by_tag(rows: Mapping[str, Mapping[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows.values():
        out[row["endpoint_tag"]] = out.get(row["endpoint_tag"], 0) + 1
    return out


def verdict_rows(decision: ProviderDecision) -> Sequence[dict[str, Any]]:
    return [asdict(v) for v in decision.verdicts]
