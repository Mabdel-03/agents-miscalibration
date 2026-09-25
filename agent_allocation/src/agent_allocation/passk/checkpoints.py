"""passk_cont_v1 in-run gates evaluated per shard from files on /orcd/data (WP-G; the conductor calls these).

``python -m agent_allocation.passk.checkpoints <stage> --config C --study-root S --shard ID [--fetch-metadata]``

* ``gate1-smoke`` -- the smoke partition: every node terminal and parsed, forced budget exact 100%, zero cache hits,
  observed model / provider == the pin; with ``--fetch-metadata`` the new ``/generation`` records must carry the
  permaslug ``qwen/qwen3.8-27b-20260814`` and the pinned provider ($0 API reads).
* ``gate2`` -- in-run checkpoint (at >= 150 settled nodes or >= $4, then every 500 nodes): exact 100%; valid-final
  pooled >= 0.90 and >= 0.85 per benchmark with n >= 20; unit cost within +-25% of the forecast for every budget with
  >= 10 calls; zero provider drift; refit forecast x 1.02 <= the shard's soft slice. Exit 4 = not due yet.
* ``gate3`` -- stop rules: an accepted node that is not exact; >= 3 ``forced_think_closed`` (or ``forced_think_short``
  with finish=stop) attempts in any 30 min window; stream cuts (``forced_think_short`` with finish None) > 2% of the
  trailing 500 attempts AND >= 10 AND spanning >= 5 min. A failing rule touches ``control/DRAIN`` of the shard.
* ``sample`` -- every 30 min: ``/generation`` of the 5 most recent results must carry the permaslug and the pin
  (a violation drains the shard).
* ``graded`` -- after the seal: every candidate (held-out included) graded (exit 4 while some are not).
* ``gate4`` -- completion: >= 99% of planned nodes succeeded per partition (failures reported, never replaced); zero
  duplicate THINK texts across legacy and new; every non-fenced candidate graded.

Exit codes: 0 pass, 3 fail (stop), 4 not due / incomplete. Records go to ``<study>/gates/<stage>_<shard>.json``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.passk import banks

EXPECTED_PERMASLUG = "qwen/qwen3.8-27b-20260814"
GATE2_FIRST_NODES = 150
GATE2_FIRST_USD = Decimal("4")
GATE2_EVERY = 500
UNIT_TOLERANCE = Decimal("0.25")
MAIN_PARTITIONS = ("c48_extra", "e1_fill", "t200_diag", "v48_bank", "t200_bank")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _shard_paths(study_root: Path, shard: str):
    from agent_allocation.paths import RunPaths
    from agent_allocation.shards import shard_run_dir
    return RunPaths.for_run(shard_run_dir(study_root / "artifacts", shard),
                            local_state_dir=study_root / "logs" / "checkpoint_local" / shard)


def _pin(paths, model: str) -> tuple[str | None, str | None]:
    data = json.loads(paths.endpoint_lock.read_text())
    lock = data["locks"].get(model) or {}
    return lock.get("provider_name"), lock.get("provider_tag")


def node_rows(paths, partitions: Sequence[str]) -> list[dict[str, Any]]:
    """One row per planned node of ``partitions`` (from the jobs manifests) with its parsed outcome, if any."""
    rows = []
    for partition in partitions:
        jobs = paths.jobs(partition)
        if not jobs.is_file():
            continue
        for line in jobs.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)["record"]
            row: dict[str, Any] = {"node_id": json.loads(line)["node_id"], "partition": partition,
                                   "item_id": rec["item_id"], "benchmark": rec["benchmark"], "kind": rec["kind"],
                                   "budget": int(rec["reasoning_budget"]), "parsed": False}
            parsed = paths.parsed(row["node_id"])
            if parsed.is_file():
                payload = json.loads(parsed.read_text())
                result = payload.get("result") or {}
                usage = result.get("usage") or {}
                cand = payload.get("candidate") or {}
                row.update(parsed=True, transport=result.get("transport_status"),
                           reasoning_tokens=usage.get("reasoning_tokens"),
                           cost=usage.get("actual_cost_usd"), cache=result.get("cache_status"),
                           model=result.get("observed_model"), provider=result.get("observed_provider"),
                           generation_id=result.get("generation_id"), has_candidate=bool(cand),
                           parse_status=cand.get("parse_status"))
                row["succeeded"] = row["transport"] == "ok" and row["has_candidate"]
                row["exact"] = row["reasoning_tokens"] == row["budget"] if row["succeeded"] else None
            rows.append(row)
    return rows


def _write(study_root: Path, name: str, record: Mapping[str, Any]) -> Path:
    target = study_root / "gates" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(record), indent=1, sort_keys=True, default=str))
    return target


# ------------------------------------------------------------------------------------------------------------------
# Gate 1 smoke
# ------------------------------------------------------------------------------------------------------------------

def gate1_smoke(cfg: Mapping[str, Any], paths, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    pin_name, pin_tag = _pin(paths, cfg["primary_model"])
    rows = node_rows(paths, ["smoke"])
    ok = [r for r in rows if r.get("succeeded")]
    crit = {
        "all_terminal": {"passed": bool(rows) and all(r["parsed"] for r in rows), "planned": len(rows),
                         "parsed": sum(r["parsed"] for r in rows), "succeeded": len(ok)},
        "exact_budgets": {"passed": bool(ok) and all(r["exact"] for r in ok),
                          "exact": sum(bool(r["exact"]) for r in ok), "of": len(ok)},
        "zero_cache_hits": {"passed": not any(str(r.get("cache") or "").upper() == "HIT" for r in rows)},
        "pin": {"passed": all(r["model"] == cfg["primary_model"] and r["provider"] == pin_name for r in ok),
                "pin": pin_name, "tag": pin_tag},
    }
    if metadata is not None:
        records = [metadata.get(r["generation_id"]) for r in ok if r.get("generation_id")]
        bad = [m for m in records if not m or m.get("model") != EXPECTED_PERMASLUG or m.get("provider_name") != pin_name]
        crit["permaslug"] = {"passed": bool(records) and not bad, "records": len(records), "violations": len(bad)}
    else:
        crit["permaslug"] = {"passed": None, "note": "not fetched (--fetch-metadata)"}
    passed = all(c["passed"] for c in crit.values() if c["passed"] is not None)
    return {"gate": "gate1_smoke", "passed": passed, "complete": crit["permaslug"]["passed"] is not None,
            "criteria": crit, "evaluated_at": _utc()}


# ------------------------------------------------------------------------------------------------------------------
# Gate 2 in-run checkpoint
# ------------------------------------------------------------------------------------------------------------------

def gate2(cfg: Mapping[str, Any], paths, *, last_checked: int = 0, force: bool = False) -> dict[str, Any]:
    from agent_allocation.live import ledger_settled
    from agent_allocation.passk.forecast import RETRY_FACTOR, UNIT_COST_USD
    shard = (cfg.get("shard") or {}).get("id")
    main_a = int(cfg["reasoning"]["main_a"])
    rows = node_rows(paths, MAIN_PARTITIONS)
    settled_rows = [r for r in rows if r["parsed"]]
    spent = ledger_settled(paths)
    due = force or (len(settled_rows) >= GATE2_FIRST_NODES or spent >= GATE2_FIRST_USD) and \
        (last_checked == 0 or len(settled_rows) - last_checked >= GATE2_EVERY)
    if not due:
        return {"gate": "gate2", "due": False, "settled_nodes": len(settled_rows), "spent": str(spent),
                "last_checked": last_checked, "evaluated_at": _utc()}
    pin_name, _ = _pin(paths, cfg["primary_model"])
    ok = [r for r in settled_rows if r.get("succeeded")]
    valid = [r for r in ok if r.get("parse_status") == "valid"]
    per_bench = {}
    for bench in sorted({r["benchmark"] for r in ok}):
        sub = [r for r in ok if r["benchmark"] == bench]
        per_bench[bench] = {"n": len(sub), "valid": sum(r.get("parse_status") == "valid" for r in sub)}
    bench_ok = all(v["valid"] / v["n"] >= 0.85 for v in per_bench.values() if v["n"] >= 20)
    units = UNIT_COST_USD.get(str(shard), {})
    by_budget: dict[int, list[Decimal]] = defaultdict(list)
    for r in ok:
        if r.get("cost") is not None:
            by_budget[r["budget"] // main_a].append(Decimal(str(r["cost"])))
    unit_rows, unit_ok = {}, True
    for mult, costs in sorted(by_budget.items()):
        mean = sum(costs, Decimal(0)) / len(costs)
        plan = units.get(mult)
        within = plan is None or len(costs) < 10 or abs(mean - plan) <= UNIT_TOLERANCE * plan
        unit_ok &= within
        unit_rows[str(mult)] = {"n": len(costs), "mean": str(mean.quantize(Decimal("0.00001"))),
                                "plan": str(plan) if plan is not None else None, "within": within}
    drift = [r["node_id"] for r in ok if r["model"] != cfg["primary_model"] or r["provider"] != pin_name]
    remaining = Counter(r["budget"] // main_a for r in rows if not r["parsed"])
    refit_unit = {m: (sum(by_budget[m], Decimal(0)) / len(by_budget[m])) if len(by_budget.get(m, [])) >= 10
                  else units.get(m, Decimal("0")) for m in set(remaining) | set(by_budget)}
    refit = spent + sum((refit_unit[m] * n for m, n in remaining.items()), Decimal(0)) * RETRY_FACTOR
    soft = Decimal(str(cfg["budget"]["soft_admission_ceiling_usd"]))
    crit = {
        "exact_budgets": {"passed": all(r["exact"] for r in ok), "exact": sum(bool(r["exact"]) for r in ok),
                          "of": len(ok)},
        "valid_final": {"passed": bool(ok) and len(valid) / len(ok) >= 0.90 and bench_ok,
                        "pooled": (len(valid) / len(ok)) if ok else None, "per_benchmark": per_bench},
        "unit_cost": {"passed": unit_ok, "by_multiplier": unit_rows},
        "provider_drift": {"passed": not drift, "drifted": drift[:20]},
        "refit_forecast": {"passed": refit <= soft, "refit_usd": str(refit.quantize(Decimal("0.01"))),
                           "soft_slice": str(soft), "spent": str(spent)},
    }
    return {"gate": "gate2", "due": True, "passed": all(c["passed"] for c in crit.values()), "criteria": crit,
            "settled_nodes": len(settled_rows), "evaluated_at": _utc(), "shard": shard}


# ------------------------------------------------------------------------------------------------------------------
# Gate 3 stop rules (incremental scan of raw attempts)
# ------------------------------------------------------------------------------------------------------------------

def _attempt_fact(path: Path) -> tuple[str, float, str | None]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        detail = ((payload.get("result") or {}).get("error_detail")) or None
    except (OSError, ValueError):
        detail = None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return str(path), mtime, detail


def attempt_index(paths) -> list[tuple[float, str | None]]:
    """``[(mtime, error_detail)]`` of every raw attempt of the shard, cached incrementally in state/."""
    from concurrent.futures import ThreadPoolExecutor
    cache_path = paths.state_dir / "passk_attempt_index.json"
    cache: dict[str, list] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except ValueError:
            cache = {}
    new = [p for p in sorted((paths.run_dir / "raw").glob("*/*.json.gz")) if str(p) not in cache]
    if new:
        with ThreadPoolExecutor(max_workers=16) as pool:
            for path, mtime, detail in pool.map(_attempt_fact, new, chunksize=32):
                cache[path] = [mtime, detail]
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, cache_path)
    return sorted((float(v[0]), v[1]) for v in cache.values())


def classify_detail(detail: str | None) -> str | None:
    if not detail:
        return None
    text = str(detail)
    if text.startswith("forced_think_closed"):
        return "closed"
    if text.startswith("forced_think_short"):
        return "stream_cut" if "finish_reason=None" in text else ("closed" if "finish_reason=stop" in text
                                                                   else "short_other")
    return None


def gate3(cfg: Mapping[str, Any], paths) -> dict[str, Any]:
    rows = node_rows(paths, [*MAIN_PARTITIONS, "smoke"])
    not_exact = [r["node_id"] for r in rows if r.get("succeeded") and r.get("exact") is False]
    attempts = attempt_index(paths)
    closed = [t for t, d in attempts if classify_detail(d) == "closed"]
    worst_window = 0
    for i, t in enumerate(closed):
        worst_window = max(worst_window, sum(1 for u in closed[i:] if u - t <= 1800))
    tail = attempts[-500:]
    cuts = [t for t, d in tail if classify_detail(d) == "stream_cut"]
    cut_rule = bool(tail) and len(cuts) > 0.02 * len(tail) and len(cuts) >= 10 and (max(cuts) - min(cuts)) >= 300
    crit = {
        "accepted_exact": {"passed": not not_exact, "violations": not_exact[:20]},
        "bias_ignored": {"passed": worst_window < 3, "max_in_30min": worst_window, "total": len(closed)},
        "stream_cuts": {"passed": not cut_rule, "trailing_attempts": len(tail), "cuts": len(cuts)},
    }
    passed = all(c["passed"] for c in crit.values())
    if not passed:
        paths.control_dir.mkdir(parents=True, exist_ok=True)
        paths.drain_sentinel.touch()
    return {"gate": "gate3", "passed": passed, "criteria": crit, "attempts": len(attempts),
            "drained": not passed, "evaluated_at": _utc()}


# ------------------------------------------------------------------------------------------------------------------
# Gate 4 completion
# ------------------------------------------------------------------------------------------------------------------

def _think_text_sha(path: Path) -> str | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        body = json.loads(payload.get("response_body_text") or "null")
        think = (body or {}).get("think") or {}
        choice = (think.get("choices") or [{}])[0]
        text = (choice.get("reasoning") or "") + (choice.get("text") or "")
        return sha256(text.encode("utf-8")).hexdigest() if text else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def think_text_shas(parsed_roots: Sequence[Path]) -> list[str]:
    """THINK text sha of every accepted result's raw exchange under the given run dirs."""
    from concurrent.futures import ThreadPoolExecutor
    raws = []
    for run_dir in parsed_roots:
        for parsed in sorted((run_dir / "parsed").glob("*/*.json")):
            rel = (json.loads(parsed.read_text()).get("result") or {}).get("raw_response_path")
            if rel:
                raws.append(run_dir / rel)
    with ThreadPoolExecutor(max_workers=32) as pool:
        return [s for s in pool.map(_think_text_sha, raws, chunksize=64) if s]


def gate4(cfg: Mapping[str, Any], paths, *, legacy_root: str | Path | None = None,
          check_duplicates: bool = True) -> dict[str, Any]:
    from agent_allocation.passk.grade_scope import heldout_splits
    rows = node_rows(paths, MAIN_PARTITIONS)
    per_partition = {}
    for partition in MAIN_PARTITIONS:
        sub = [r for r in rows if r["partition"] == partition]
        if not sub:
            continue
        done = sum(bool(r.get("succeeded")) for r in sub)
        per_partition[partition] = {"planned": len(sub), "succeeded": done, "rate": done / len(sub),
                                    "passed": done / len(sub) >= 0.99,
                                    "failed": [r["node_id"] for r in sub if r["parsed"] and not r.get("succeeded")][:50]}
    crit: dict[str, Any] = {"completion": {"passed": bool(per_partition) and all(v["passed"] for v in
                                                                               per_partition.values()),
                                           "partitions": per_partition}}
    # "dev grades complete": the non-held-out partitions, whatever the seal state (held-out grading is `graded`)
    heldout = {p for ps in heldout_splits(cfg).values() for p in ps}
    ungraded = [r["node_id"] for r in rows if r.get("has_candidate") and r["partition"] not in heldout
                and not paths.grade(r["node_id"]).exists()]
    crit["dev_grades"] = {"passed": not ungraded, "ungraded": len(ungraded)}
    if check_duplicates:
        roots = [paths.run_dir] + ([Path(legacy_root) / "shards" / s for s in ("mancer", "parasail")]
                                   if legacy_root else [])
        shas = think_text_shas(roots)
        dupes = [s for s, n in Counter(shas).items() if n > 1]
        crit["unique_think_texts"] = {"passed": not dupes, "texts": len(shas), "duplicates": len(dupes)}
    # per held-out split: its own partitions complete + every shared criterion (the non-held-out partitions complete,
    # dev grades, unique THINK texts). V48 is released once v48_bank passes, T200 once t200_bank passes.
    shared_ok = all(c["passed"] for k, c in crit.items() if k != "completion") and all(
        v["passed"] for p, v in per_partition.items() if p not in heldout)
    splits = {}
    for split, parts in heldout_splits(cfg).items():
        mine = [per_partition[p] for p in parts if p in per_partition]
        splits[split] = {"passed": bool(per_partition) and shared_ok and all(v["passed"] for v in mine),
                         "partitions": [p for p in parts if p in per_partition]}
    return {"gate": "gate4", "passed": all(c["passed"] for c in crit.values()), "criteria": crit,
            "splits": splits, "evaluated_at": _utc()}


def graded(cfg: Mapping[str, Any], paths) -> dict[str, Any]:
    """After the seal: every candidate the fence now allows (held-out included) has a grade."""
    from agent_allocation.passk.grade_scope import grade_scope
    skip, only = grade_scope(paths, cfg)
    rows = node_rows(paths, MAIN_PARTITIONS)
    ungraded = [r["node_id"] for r in rows if r.get("has_candidate") and not paths.grade(r["node_id"]).exists()]
    fenced = [n for n in ungraded if n in skip]
    return {"gate": "graded", "passed": not ungraded, "ungraded": len(ungraded), "still_fenced": len(fenced),
            "evaluated_at": _utc()}


# ------------------------------------------------------------------------------------------------------------------
# Gate 3 sampling: permaslug + provider of the most recent generations (every 30 min; $0 /generation reads)
# ------------------------------------------------------------------------------------------------------------------

SAMPLE_MIN_AGE_S = 300.0     # /generation indexes a result with a lag: never judge a generation younger than this
SAMPLE_PENDING_CAP = 25      # unavailable ids carried to the next sample (retried later, never counted as drift)


def recent_generation_ids(paths, n: int = 5, *, min_age_s: float = SAMPLE_MIN_AGE_S,
                          now: float | None = None) -> list[str]:
    import time
    cutoff = (time.time() if now is None else now) - min_age_s
    files = [p for p in (paths.run_dir / "parsed").glob("*/*.json") if p.stat().st_mtime <= cutoff]
    parsed = sorted(files, key=lambda p: p.stat().st_mtime)[-n:]
    out = []
    for path in parsed:
        gid = (json.loads(path.read_text()).get("result") or {}).get("generation_id")
        if gid:
            out.append(gid)
    return out


def sample_check(cfg: Mapping[str, Any], paths, metadata: Mapping[str, Any],
                 requested: list[str] | None = None) -> dict[str, Any]:
    """Permaslug + provider of sampled generations. Only a POSITIVE mismatch (a fetched record naming another
    permaslug or another provider) is drift. An id whose /generation record is not available yet (404 -> absent,
    None, or a record with neither model nor provider) is `unavailable`: recorded and retried by the next sample,
    never drift. `passed` needs at least one conclusive record and no drift; `drift` alone stops the run."""
    pin_name, _ = _pin(paths, cfg["primary_model"])
    rows, unavailable = [], []
    for gid in (requested if requested is not None else list(metadata)):
        data = metadata.get(gid)
        data = data.get("data", data) if isinstance(data, Mapping) else None
        if not isinstance(data, Mapping):
            unavailable.append(gid)
            continue
        slugs = ({data.get("model")} | {r.get("model_permaslug") for r in data.get("provider_responses") or []
                                        if isinstance(r, Mapping)}) - {None, ""}
        provider = data.get("provider_name")
        if not slugs and not provider:
            unavailable.append(gid)
            continue
        slug_ok = not slugs or slugs == {EXPECTED_PERMASLUG}
        provider_ok = not provider or provider == pin_name
        rows.append({"generation_id": gid, "ok": slug_ok and provider_ok, "slugs": sorted(map(str, slugs)),
                     "provider": provider})
    drift = any(not r["ok"] for r in rows)
    return {"gate": "gate3_sample", "passed": bool(rows) and not drift, "drift": drift, "conclusive": bool(rows),
            "samples": rows, "unavailable": unavailable[:SAMPLE_PENDING_CAP], "evaluated_at": _utc()}


# ------------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_allocation.passk.checkpoints")
    parser.add_argument("stage", choices=["gate1-smoke", "gate2", "gate3", "gate4", "sample", "graded"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--fetch-metadata", action="store_true", help="gate1-smoke: read /generation ($0, live)")
    parser.add_argument("--force", action="store_true", help="gate2: evaluate even if not due")
    parser.add_argument("--split", choices=("v48", "t200"), default=None,
                        help="gate4: pass/fail on this held-out split only (its partitions + the shared criteria)")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    from agent_allocation.shards import shard_config
    study = Path(args.study_root)
    cfg = shard_config(load_config(args.config), args.shard)
    paths = _shard_paths(study, args.shard)
    if args.stage == "gate1-smoke":
        metadata = None
        if args.fetch_metadata:
            import asyncio
            from agent_allocation.live import fetch_metadata
            ids = [r["generation_id"] for r in node_rows(paths, ["smoke"]) if r.get("generation_id")]
            metadata = {k: (v.get("data") if isinstance(v, Mapping) and "data" in v else v)
                        for k, v in (asyncio.run(fetch_metadata(paths, ids)) or {}).items()}
        result = gate1_smoke(cfg, paths, metadata=metadata)
        name, code = f"gate1_smoke_{args.shard}", (0 if result["passed"] and result["complete"] else 3)
    elif args.stage == "gate2":
        state = study / "gates" / f"gate2_{args.shard}.json"
        last = json.loads(state.read_text()).get("settled_nodes", 0) if state.is_file() else 0
        result = gate2(cfg, paths, last_checked=last, force=args.force)
        if not result["due"]:
            print(json.dumps(result, indent=1))
            return 4
        name, code = f"gate2_{args.shard}", (0 if result["passed"] else 3)
    elif args.stage == "graded":
        result = graded(cfg, paths)
        name, code = f"graded_{args.shard}", (0 if result["passed"] else 4)
    elif args.stage == "sample":
        import asyncio
        from agent_allocation.live import fetch_metadata
        prior = study / "gates" / f"gate3_sample_{args.shard}.json"
        try:
            pending = list(json.loads(prior.read_text()).get("unavailable") or []) if prior.is_file() else []
        except (OSError, ValueError):
            pending = []
        ids = list(dict.fromkeys([*pending, *recent_generation_ids(paths)]))[:SAMPLE_PENDING_CAP]
        if not ids:
            print(json.dumps({"gate": "gate3_sample", "due": False}))
            return 4
        result = sample_check(cfg, paths, asyncio.run(fetch_metadata(paths, ids, attempts=6, delay_s=20.0)) or {},
                              requested=ids)
        # 3 only on a positive permaslug/provider mismatch; nothing fetched yet = inconclusive (4), retried next time
        name, code = f"gate3_sample_{args.shard}", (3 if result["drift"] else 0 if result["passed"] else 4)
        if result["drift"]:
            paths.control_dir.mkdir(parents=True, exist_ok=True)
            paths.drain_sentinel.touch()
    elif args.stage == "gate3":
        result = gate3(cfg, paths)
        name, code = f"gate3_{args.shard}", (0 if result["passed"] else 3)
    else:
        result = gate4(cfg, paths, legacy_root=cfg["passk"]["legacy_run_root"])
        if args.split:
            ok = bool((result["splits"].get(args.split) or {}).get("passed"))
            result = {**result, "split": args.split, "passed": ok}
            name, code = f"gate4_{args.shard}_{args.split}", (0 if ok else 3)
        else:
            name, code = f"gate4_{args.shard}", (0 if result["passed"] else 3)
    target = _write(study, name, result)
    print(json.dumps({"gate": name, "record": str(target), "passed": result.get("passed")}, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
