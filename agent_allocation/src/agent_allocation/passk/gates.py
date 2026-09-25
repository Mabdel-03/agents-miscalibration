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


# ------------------------------------------------------------------------------------------------------------------
# Gate 0 (offline + optional live /key re-read)
# ------------------------------------------------------------------------------------------------------------------

GATE0_TESTS = ("tests/unit/passk", "tests/integration/test_passk_wpg_e2e.py")
HOME_PREFIXES = ("/home/",)


def _legacy_decoding_hash(legacy_root: Path) -> str | None:
    for jobs in sorted(legacy_root.glob("shards/*/manifests/jobs/main.jsonl")):
        with open(jobs, encoding="utf-8") as handle:
            line = handle.readline()
        if line.strip():
            return json.loads(line)["record"]["decoding_config_hash"]
    return None


def _run_tests(code_dir: Path, targets: Sequence[str]) -> dict[str, Any]:
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    proc = subprocess.run(cmd, cwd=code_dir, capture_output=True, text=True, check=False)
    tail = (proc.stdout or "").strip().splitlines()[-3:]
    return {"passed": proc.returncode == 0, "returncode": proc.returncode, "targets": list(targets), "tail": tail}


def _git_head(code_dir: Path) -> dict[str, Any]:
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=code_dir,
                           capture_output=True, text=True).stdout.strip()
    return {"head": head, "clean": dirty == ""}


def key_check(base_cfg: Mapping[str, Any], snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Sum of shard credit ceilings <= limit_remaining - 3 (the owner's key limit covers the whole cap)."""
    from agent_allocation.shards import shard_config, shard_ids
    total = sum(Decimal(str(shard_config(base_cfg, s)["budget"]["credit_ceiling_usd"])) for s in shard_ids(base_cfg))
    if snapshot is None:
        return {"passed": None, "note": "not evaluated (offline)", "sum_credit_ceilings": str(total)}
    remaining = snapshot.get("key", {}).get("limit_remaining")
    if remaining is None:
        return {"passed": False, "note": "key has no limit_remaining", "sum_credit_ceilings": str(total)}
    ok = total <= Decimal(str(remaining)) - Decimal("3")
    return {"passed": ok, "sum_credit_ceilings": str(total), "limit_remaining": str(remaining)}


def evaluate_gate0(base_cfg: Mapping[str, Any], study_root: str | Path, *, legacy_root: str | Path | None = None,
                   snapshot: Mapping[str, Any] | None = None, tests: dict[str, Any] | None = None,
                   e0: Mapping[str, Any] | None = None, local_state_root: str | Path | None = None
                   ) -> dict[str, Any]:
    """Every offline Gate 0 criterion of the plan over the materialized shards of ``study_root``."""
    import os
    from agent_allocation.live import load_tasks, read_endpoint_locks
    from agent_allocation.passk import banks
    from agent_allocation.passk.forecast import forecast_study
    from agent_allocation.passk.run import shard_plan
    from agent_allocation.paths import RunPaths
    from agent_allocation.prompts.render import prompt_version
    from agent_allocation.config import decoding_config_hash
    from agent_allocation.shards import shard_config, shard_ids, shard_run_dir
    study = Path(study_root)
    artifacts = study / "artifacts"
    legacy_root = Path(legacy_root or base_cfg["passk"]["legacy_run_root"])
    crit: dict[str, Any] = {}
    # Gate S and the splits manifest
    gate_s = study / "gates" / "gate_s.json"
    splits_sha = banks.splits_sha256(artifacts) if banks.splits_path(artifacts).is_file() else None
    if gate_s.is_file():
        rec = json.loads(gate_s.read_text())
        recorded = rec.get("splits_sha256") or (rec.get("sha256") or {}).get("splits.json")
        crit["gate_s"] = {"passed": bool(rec.get("passed")) and (recorded is None or recorded == splits_sha),
                          "recorded_splits_sha256": recorded, "splits_sha256": splits_sha}
    else:
        crit["gate_s"] = {"passed": False, "note": f"{gate_s} missing", "splits_sha256": splits_sha}
    splits = banks.load_splits(artifacts)
    plans, locks, tasks_all = {}, {}, {}
    for sid in shard_ids(base_cfg):
        cfg = shard_config(base_cfg, sid)
        local = Path(local_state_root) / sid if local_state_root else None
        paths = RunPaths.for_run(shard_run_dir(artifacts, sid), local_state_dir=local or
                                 (study / "logs" / "gate0_local" / sid))
        shard_locks = read_endpoint_locks(paths)
        if cfg["primary_model"] not in shard_locks:
            crit[f"lock_{sid}"] = {"passed": False, "note": "no endpoint lock (passk-run --adopt-legacy-lock)"}
            continue
        locks[sid] = shard_locks[cfg["primary_model"]]
        tasks = load_tasks(paths)
        tasks_all.update(tasks)
        plans[sid] = shard_plan(cfg, paths, banks.PARTITIONS, splits=splits, tasks=tasks, lock=locks[sid])
    counts = {p: sum(len([n for g in plans[s].get(p, []) for n in g.nodes]) for s in plans) for p in banks.PARTITIONS}
    all_ids = [n.node_id for s in plans for gs in plans[s].values() for g in gs for n in g.nodes]
    crit["planned_counts"] = {"passed": counts == banks.EXPECTED_COUNTS, "counts": counts,
                              "expected": banks.EXPECTED_COUNTS}
    legacy_ids = banks.legacy_node_ids(legacy_root) if legacy_root.is_dir() else set()
    crit["node_ids"] = {"passed": len(all_ids) == len(set(all_ids)) == sum(banks.EXPECTED_COUNTS.values())
                        and bool(legacy_ids) and not (set(all_ids) & legacy_ids),
                        "planned": len(all_ids), "unique": len(set(all_ids)), "legacy_ids": len(legacy_ids),
                        "legacy_collisions": len(set(all_ids) & legacy_ids)}
    legacy_dec = _legacy_decoding_hash(legacy_root)
    crit["prompt_and_decoding"] = {
        "passed": prompt_version() == base_cfg["passk"]["legacy_prompt_version"]
        and decoding_config_hash(base_cfg) == legacy_dec,
        "prompt_version": prompt_version(), "decoding_config_hash": decoding_config_hash(base_cfg),
        "legacy_decoding_config_hash": legacy_dec}
    parity_items = [*splits["panels"]["L96"], *splits["panels"]["D32"]]
    crit["think_parity"] = think_parity(base_cfg, legacy_root, parity_items, tasks_all, locks=locks) \
        if len(locks) == len(shard_ids(base_cfg)) else {"passed": False, "note": "locks missing"}
    e0 = e0 if e0 is not None else (json.loads((study / "gates" / "e0_gate.json").read_text())
                                    if (study / "gates" / "e0_gate.json").is_file() else None)
    crit["e0_gate"] = {"passed": bool(e0 and e0.get("passed")),
                       "checks": {k: v.get("passed") for k, v in ((e0 or {}).get("checks") or {}).items()}}
    crit["slices"] = slices_match(base_cfg)
    slices = {s: shard_config(base_cfg, s)["budget"]["soft_admission_ceiling_usd"] for s in shard_ids(base_cfg)}
    forecast = forecast_study(plans, max_usd=base_cfg["passk"].get("forecast_max_usd", 115), slices=slices)
    crit["forecast"] = {"passed": bool(forecast["passed"]), "total_usd": forecast["total_usd"],
                        "by_shard": {s: v["forecast_usd"] for s, v in forecast["shards"].items()},
                        "checks": forecast["checks"]}
    crit["key"] = key_check(base_cfg, snapshot)
    home = [k for k in ("UV_CACHE_DIR", "HF_HOME", "MPLCONFIGDIR", "TMPDIR")
            if str(os.environ.get(k, "")).startswith(HOME_PREFIXES)]
    crit["nothing_under_home"] = {"passed": not home and not str(study.resolve()).startswith(HOME_PREFIXES),
                                  "env_under_home": home}
    crit["tests"] = tests or {"passed": None, "note": "not run (--run-tests)"}
    required = [k for k in crit if k not in ("key", "tests")]
    passed = all(crit[k].get("passed") for k in required) and crit["key"].get("passed") is not False \
        and crit["tests"].get("passed") is not False
    return {"gate": "gate0", "passed": bool(passed), "criteria": crit, "jobs_sha256": jobs_shas(plans),
            "splits_sha256": splits_sha, "evaluated_at": _utc(),
            "complete": crit["key"].get("passed") is not None and crit["tests"].get("passed") is not None}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="python -m agent_allocation.passk.gates")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g0 = sub.add_parser("gate0", help="evaluate Gate 0 and write <study>/gates/gate0.json")
    g0.add_argument("--config", required=True)
    g0.add_argument("--study-root", required=True)
    g0.add_argument("--check-key", action="store_true", help="re-read /key (live, $0) for the sum-of-slices check")
    g0.add_argument("--run-tests", action="store_true", help="run the passk unit suite + mocked e2e first")
    g0.add_argument("--run-e0", action="store_true", help="(re)run the E0 gate subset (several minutes)")
    g0.add_argument("--overwrite", action="store_true", help="allow a changed jobs sha (owner decision only)")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    from agent_allocation.shards import base_config
    cfg = base_config(load_config(args.config))
    study = Path(args.study_root)
    code_dir = Path(__file__).resolve().parents[3]
    tests = _run_tests(code_dir, GATE0_TESTS) if args.run_tests else None
    e0 = None
    if args.run_e0:
        from agent_allocation.passk.e0_gate import e0_gate, write_e0_gate
        from agent_allocation.passk.banks import load_splits
        e0 = e0_gate(cfg["passk"]["legacy_run_root"], splits=load_splits(study / "artifacts"))
        write_e0_gate(study, e0)
    snapshot = None
    if args.check_key:
        from agent_allocation.apikey import load_api_key
        from agent_allocation.doctor import provider_snapshot as fetch
        snapshot = fetch(load_api_key())
    result = evaluate_gate0(cfg, study, snapshot=snapshot, tests=tests, e0=e0)
    result["code"] = _git_head(code_dir)
    target = write_gate0(study, result, overwrite=args.overwrite)
    print(json.dumps({"gate0": str(target), "passed": result["passed"], "complete": result["complete"],
                      "criteria": {k: v.get("passed") for k, v in result["criteria"].items()}}, indent=1))
    return 0 if result["passed"] and result["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
