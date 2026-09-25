"""passk_cont_v1 Gate 0 pieces (WP-G): THINK-body parity with forced_v2 and the Gate 0 record.

THINK parity (plan, Gate 0): for every legacy L96/D32 item (C48/V48 are L96 items), the THINK body the new harness
would send for a node at ``a = A`` (and a DEEP node at ``2A``: DEEP renders ``root.txt`` exactly like ROOT) --
``ForcedClient.validate_request`` over ``render_messages_with_report`` -- must equal the legacy raw
``request_body.think`` of that item's legacy ROOT node at ``A`` apart from ``max_tokens``, with the same
``provider.only`` (the legacy pinned tag). Offline: needs only an endpoint lock (the legacy shard lock in tests).

The Gate 0 record (``<study>/gates/gate0.json``) carries the per-shard jobs sha256 of every partition that
``passk-run`` asserts before any dispatch.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

LEGACY_PARTITION_PREFERENCE = ("main", "calibration", "pilot", "canary")
IGNORED_KEYS = ("max_tokens",)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_raw(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def legacy_root_nodes(legacy_shard_dir: Path, main_a: int = 1024) -> dict[str, list[str]]:
    """``{item_id: [node ids]}`` of legacy forced ROOT nodes at ``A`` (slot 0 first), by partition preference."""
    out: dict[str, list[tuple[int, int, str]]] = {}
    for rank, partition in enumerate(LEGACY_PARTITION_PREFERENCE):
        jobs = legacy_shard_dir / "manifests" / "jobs" / f"{partition}.jsonl"
        if not jobs.is_file():
            continue
        with open(jobs, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rec = row["record"]
                if rec.get("kind") == "root" and int(rec.get("reasoning_budget", 0)) == main_a:
                    out.setdefault(rec["item_id"], []).append((rank, int(rec.get("root_slot", 0)), row["node_id"]))
    return {item: [n for _, _, n in sorted(v)] for item, v in out.items()}


def legacy_think_body(legacy_shard_dir: Path, node_ids: Sequence[str], max_attempts: int = 4) -> dict[str, Any] | None:
    """The first legacy raw ``request_body.think`` found for any of ``node_ids`` (attempts 1..max)."""
    from agent_allocation.paths import RunPaths
    paths = RunPaths.for_run(legacy_shard_dir, local_state_dir=legacy_shard_dir / "state")   # path math only
    for node_id in node_ids:
        for attempt in range(1, max_attempts + 1):
            raw = paths.raw_response(f"{node_id}:{attempt}")
            if raw.is_file():
                payload = _read_raw(raw)
                body = ((payload or {}).get("request_body") or {}).get("think")
                if isinstance(body, Mapping):
                    return dict(body)
    return None


def new_think_body(cfg: Mapping[str, Any], task, lock, *, kind: str = "root", budget: int | None = None,
                   partition: str = "parity", client=None) -> dict[str, Any]:
    from agent_allocation.api.fake import fake_openrouter_transport
    from agent_allocation.api.forced import make_generation_client
    from agent_allocation.contracts import GenerationKind
    from agent_allocation.live import standalone_spec
    from agent_allocation.prompts.render import render_messages_with_report
    budget = int(cfg["reasoning"]["main_a"]) if budget is None else int(budget)
    spec = standalone_spec(cfg=cfg, partition=partition, task=task, lock=lock, kind=GenerationKind(kind),
                           budget=budget, root_slot=0, draw_id=0, amendments={})
    messages, _report = render_messages_with_report(spec, task, [], seed=int(cfg["seed"]))
    own = client is None
    client = client or make_generation_client(cfg, api_key=None, transport=fake_openrouter_transport())
    return dict(client.validate_request(spec, messages, lock, dict(cfg.get("sampling", {}))))


def _strip(body: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k not in IGNORED_KEYS}


def think_parity(base_cfg: Mapping[str, Any], legacy_root: str | Path, items: Sequence[str],
                 tasks: Mapping[str, Any], *, locks: Mapping[str, Any] | None = None,
                 assignment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Compare new vs legacy THINK bodies for ``items`` (each on its legacy shard). ``locks``: ``{shard: lock}``
    (default: each legacy shard's own lock). ``assignment``: ``{item: shard}`` (default: legacy assignment.json)."""
    from agent_allocation.api.fake import fake_openrouter_transport
    from agent_allocation.api.forced import make_generation_client
    from agent_allocation.contracts import EndpointLock, load_record
    from agent_allocation.shards import shard_config
    legacy_root = Path(legacy_root)
    if assignment is None:
        assignment = json.loads((legacy_root / "shards" / "assignment.json").read_text())["assignment"]
    shard_cache: dict[str, Any] = {}
    results = []
    mismatches, missing = [], []
    provider_only: dict[str, list] = {}
    for item in items:
        shard = assignment.get(item)
        if shard is None:
            missing.append({"item_id": item, "reason": "no legacy shard"})
            continue
        if shard not in shard_cache:
            shard_dir = legacy_root / "shards" / shard
            cfg = shard_config(base_cfg, shard)
            if locks and shard in locks:
                lock = locks[shard]
            else:
                data = json.loads((shard_dir / "locks" / "openrouter_endpoints.json").read_text())
                lock = load_record(EndpointLock, data["locks"][cfg["primary_model"]])
            client = make_generation_client(cfg, api_key=None, transport=fake_openrouter_transport())
            shard_cache[shard] = (shard_dir, cfg, lock, client, legacy_root_nodes(shard_dir,
                                                                                 int(cfg["reasoning"]["main_a"])))
        shard_dir, cfg, lock, client, roots = shard_cache[shard]
        legacy = legacy_think_body(shard_dir, roots.get(item, []))
        if legacy is None:
            missing.append({"item_id": item, "shard": shard, "reason": "no legacy raw THINK body"})
            continue
        task = tasks.get(item)
        if task is None:
            missing.append({"item_id": item, "shard": shard, "reason": "no public task"})
            continue
        main_a = int(cfg["reasoning"]["main_a"])
        root_body = new_think_body(cfg, task, lock, kind="root", budget=main_a, client=client)
        deep_body = new_think_body(cfg, task, lock, kind="deep", budget=2 * main_a, client=client)
        tag = cfg["provider"].get("endpoint_tag")
        ok_root = _strip(root_body) == _strip(legacy)
        ok_deep = _strip(deep_body) == _strip(legacy)
        only_new = (root_body.get("provider") or {}).get("only")
        only_legacy = (legacy.get("provider") or {}).get("only")
        ok_provider = only_new == only_legacy == [tag]
        ok_budget = root_body.get("max_tokens") == main_a and deep_body.get("max_tokens") == 2 * main_a
        provider_only.setdefault(shard, only_legacy)
        row = {"item_id": item, "shard": shard, "root_equal": ok_root, "deep_equal": ok_deep,
               "provider_only_equal": ok_provider, "max_tokens_ok": ok_budget}
        results.append(row)
        if not (ok_root and ok_deep and ok_provider and ok_budget):
            diff = sorted(k for k in set(root_body) | set(legacy)
                          if k not in IGNORED_KEYS and root_body.get(k) != legacy.get(k))
            mismatches.append({**row, "differing_keys": diff})
    passed = bool(results) and not mismatches and not missing
    return {"check": "think_body_parity", "passed": passed, "checked": len(results), "items": len(items),
            "mismatches": mismatches, "missing": missing, "provider_only": provider_only, "evaluated_at": _utc()}


# ------------------------------------------------------------------------------------------------------------------
# Gate 0 record
# ------------------------------------------------------------------------------------------------------------------

def gate0_path(study_root: str | Path) -> Path:
    return Path(study_root) / "gates" / "gate0.json"


def jobs_shas(shard_plans: Mapping[str, Mapping[str, Sequence[Any]]]) -> dict[str, dict[str, str]]:
    from agent_allocation.passk.banks import jobs_text_sha
    return {shard: {p: jobs_text_sha(gs) for p, gs in plan.items()} for shard, plan in shard_plans.items()}


def write_gate0(study_root: str | Path, record: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    """Write ``gates/gate0.json``. Refuses to change a recorded jobs sha (the frozen lists) unless ``overwrite``."""
    target = gate0_path(study_root)
    if target.is_file() and not overwrite:
        previous = json.loads(target.read_text())
        if previous.get("jobs_sha256") and previous["jobs_sha256"] != record.get("jobs_sha256"):
            raise RuntimeError(f"{target}: recorded jobs sha256 differ from the new plan; refusing to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(record), indent=1, sort_keys=True, default=str))
    return target


def slices_match(base_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Per-shard credit/soft slices of the derived configs equal ``budget_by_shard`` and sum to the base budget."""
    from agent_allocation.shards import shard_config, shard_ids
    rows, ok = {}, True
    for sid in shard_ids(base_cfg):
        derived = shard_config(base_cfg, sid)["budget"]
        want = (base_cfg.get("budget_by_shard") or {}).get(sid) or {}
        row = {k: str(Decimal(str(derived[k])).quantize(Decimal("0.01"))) for k in ("credit_ceiling_usd",
                                                                            "soft_admission_ceiling_usd")}
        ok &= all(Decimal(str(derived[k])) == Decimal(str(v)) for k, v in want.items())
        rows[sid] = row
    for key in ("credit_ceiling_usd", "soft_admission_ceiling_usd"):
        total = sum(Decimal(r[key]) for r in rows.values())
        ok &= total <= Decimal(str(base_cfg["budget"][key]))
    return {"passed": bool(ok), "slices": rows}
