"""E0 gate subset (offline, $0; blocks launch) over the READ-ONLY forced_v2 run root (WP-G).

Checks (plan, "E0 -- Gate subset"):

1. ``generation_ids_unique`` -- every THINK and ANSWER generation id of every accepted legacy result (``parsed/``
   results, THINK id from the result's own raw payload) is unique across both shards (a natural calibration
   exchange is one call, so its THINK and ANSWER ids coincide and count once).
2. ``request_body_collisions`` -- THINK request bodies (canonical JSON, ``max_tokens`` included) hashed over every
   raw exchange; a hash shared by distinct nodes is expected only within one item's context-free bank (ROOT/DEEP
   nodes without parents: no seed is sent, so replicas are byte-identical by design). Any other shared hash must be
   one of the known benign private-chain pairs (:data:`KNOWN_BENIGN_ITEMS`): exactly those, no more.
3. ``observed_model_provider`` -- ``observed_model == qwen/qwen3.8-27b`` and ``observed_provider`` equals the shard's
   pinned provider (its endpoint lock's ``provider_name``) on every parsed result.
4. ``generation_metadata`` -- every record of ``shards/*/state/generation_metadata.jsonl`` (1,054) carries permaslug
   ``qwen/qwen3.8-27b-20260814`` (``data.model`` and every ``provider_responses[].model_permaslug``) and
   ``provider_name`` equal to the shard pin.
5. ``forced_budget_check`` -- ``merged/tables/forced_budget_check.parquet``: exact == generations, short == long == 0
   on every row.
6. ``split_overlap`` (when a splits manifest is given) -- T200 disjoint from L96 and D32; C48 = V48 within L96.

Writes ``<study>/gates/e0_gate.json``. Nothing is written under the legacy root (files are opened read-only).
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

EXPECTED_MODEL = "qwen/qwen3.8-27b"
EXPECTED_PERMASLUG = "qwen/qwen3.8-27b-20260814"
EXPECTED_METADATA_RECORDS = 1054
KNOWN_BENIGN_ITEMS = ("omni_math_hard/3907", "hmmt/2026/24")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _read_gz(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _shards(legacy_root: Path) -> list[Path]:
    return sorted(p for p in (legacy_root / "shards").iterdir() if p.is_dir())


def _pin(shard_dir: Path) -> tuple[str | None, str | None]:
    data = json.loads((shard_dir / "locks" / "openrouter_endpoints.json").read_text())
    lock = data["locks"].get(EXPECTED_MODEL) or {}
    return lock.get("provider_name"), lock.get("provider_tag")


def _specs(shard_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for jobs in sorted((shard_dir / "manifests" / "jobs").glob("*.jsonl")):
        with open(jobs, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    out[row["node_id"]] = row["record"]
    return out


def _think_answer_ids(payload: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    from agent_allocation.api.forced import forced_generation_ids
    if not payload:
        return None, None
    text = payload.get("response_body_text")
    try:
        body = json.loads(text) if isinstance(text, str) else None
    except ValueError:
        body = None
    if body is None:
        return None, None
    try:
        return forced_generation_ids(body)
    except Exception:  # noqa: BLE001 - a malformed legacy body is reported, not fatal
        return None, None


def _raw_facts(path: Path) -> tuple[str, str | None, str | None, str | None, str | None]:
    """(path, node_id, THINK body sha, THINK id, ANSWER id) of one raw exchange."""
    payload = _read_gz(path)
    if not payload:
        return str(path), None, None, None, None
    think = (payload.get("request_body") or {}).get("think")
    digest = sha256(_canonical(think)).hexdigest() if isinstance(think, Mapping) else None
    think_id, answer_id = _think_answer_ids(payload)
    return str(path), str(payload.get("node_id")), digest, think_id, answer_id


def _parsed_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())["result"]


def scan_legacy(legacy_root: str | Path, *, threads: int = 32) -> dict[str, Any]:
    """One pass over parsed/ and raw/ of every legacy shard (I/O-bound on NFS: a thread pool reads the files);
    returns the raw material of checks 1-3."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(legacy_root)
    gen_ids: list[str] = []
    missing_think = []
    model_bad, provider_bad = [], []
    body_nodes: dict[str, set[str]] = defaultdict(set)
    specs_all: dict[str, dict[str, Any]] = {}
    parsed_count = raw_count = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for shard_dir in _shards(root):
            pin_name, _tag = _pin(shard_dir)
            specs_all.update(_specs(shard_dir))
            raw_paths = sorted((shard_dir / "raw").glob("*/*.json.gz"))
            facts = {}
            for path, node_id, digest, think_id, answer_id in pool.map(_raw_facts, raw_paths, chunksize=64):
                facts[path] = (think_id, answer_id)
                if digest is not None:
                    raw_count += 1
                    body_nodes[digest].add(node_id)
            parsed_paths = sorted((shard_dir / "parsed").glob("*/*.json"))
            for result in pool.map(_parsed_result, parsed_paths, chunksize=64):
                parsed_count += 1
                if result.get("observed_model") != EXPECTED_MODEL:
                    model_bad.append((shard_dir.name, result.get("node_id"), result.get("observed_model")))
                if result.get("observed_provider") != pin_name:
                    provider_bad.append((shard_dir.name, result.get("node_id"), result.get("observed_provider")))
                answer_id = result.get("generation_id")
                think_id = None
                raw_rel = result.get("raw_response_path")
                if raw_rel:
                    think_id, raw_answer = facts.get(str(shard_dir / raw_rel), (None, None))
                    answer_id = answer_id or raw_answer
                if think_id is None:
                    missing_think.append(result.get("node_id"))
                # a NATURAL (unforced calibration) exchange is one call: its THINK and ANSWER ids coincide
                gen_ids.extend(sorted({g for g in (think_id, answer_id) if g}))
    return {"gen_ids": gen_ids, "missing_think": missing_think, "model_bad": model_bad, "provider_bad": provider_bad,
            "body_nodes": body_nodes, "specs": specs_all, "parsed": parsed_count, "raw_think_bodies": raw_count}


def classify_collisions(body_nodes: Mapping[str, set[str]], specs: Mapping[str, Mapping[str, Any]]
                        ) -> tuple[list[dict[str, Any]], int]:
    """(unexpected collision groups, number of by-design bank groups)."""
    unexpected, by_design = [], 0
    for digest, nodes in body_nodes.items():
        if len(nodes) < 2:
            continue
        recs = [specs.get(n) for n in sorted(nodes)]
        items = {r.get("item_id") for r in recs if r}
        context_free = all(r is not None and r.get("kind") in ("root", "deep") and not r.get("parent_ids")
                           for r in recs)
        if context_free and len(items) == 1:
            by_design += 1
            continue
        unexpected.append({"sha256": digest, "nodes": sorted(nodes), "items": sorted(i for i in items if i),
                           "kinds": sorted({r.get("kind") for r in recs if r}),
                           "rounds": sorted({(r.get("root_slot"), r.get("update_round"), r.get("team_size"))
                                             for r in recs if r})})
    return unexpected, by_design


def check_metadata(legacy_root: str | Path) -> dict[str, Any]:
    root = Path(legacy_root)
    total, bad = 0, []
    for shard_dir in _shards(root):
        pin_name, _ = _pin(shard_dir)
        source = shard_dir / "state" / "generation_metadata.jsonl"
        if not source.is_file():
            continue
        with open(source, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                data = json.loads(line).get("data") or {}
                slugs = {data.get("model")} | {r.get("model_permaslug") for r in data.get("provider_responses") or []}
                if slugs != {EXPECTED_PERMASLUG} or data.get("provider_name") != pin_name:
                    bad.append({"shard": shard_dir.name, "id": data.get("id"), "slugs": sorted(map(str, slugs)),
                                "provider": data.get("provider_name")})
    return {"passed": total == EXPECTED_METADATA_RECORDS and not bad, "records": total,
            "expected_records": EXPECTED_METADATA_RECORDS, "violations": bad[:20], "violation_count": len(bad)}


def check_forced_budget(legacy_root: str | Path) -> dict[str, Any]:
    import pandas as pd
    table = pd.read_parquet(Path(legacy_root) / "merged" / "tables" / "forced_budget_check.parquet")
    bad = table[(table["exact"] != table["generations"]) | (table["short"] != 0) | (table["long"] != 0)]
    return {"passed": len(table) > 0 and bad.empty, "rows": int(len(table)),
            "generations": int(table["generations"].sum()), "violations": bad.to_dict(orient="records")}


def check_splits(splits: Mapping[str, Any] | None) -> dict[str, Any]:
    if splits is None:
        return {"passed": None, "note": "no splits manifest given"}
    p = {k: set(v) for k, v in splits["panels"].items()}
    overlap = sorted(p["T200"] & (p["L96"] | p["D32"]))
    ok = not overlap and p["C48"] == p["V48"] and p["C48"] <= p["L96"] and not (p["L96"] & p["D32"])
    return {"passed": ok, "t200_overlap": overlap, "c48_eq_v48": p["C48"] == p["V48"],
            "c48_in_l96": p["C48"] <= p["L96"], "l96_d32_overlap": sorted(p["L96"] & p["D32"])}


def e0_gate(legacy_root: str | Path, *, splits: Mapping[str, Any] | None = None,
            known_benign_items: Iterable[str] = KNOWN_BENIGN_ITEMS) -> dict[str, Any]:
    scan = scan_legacy(legacy_root)
    ids = scan["gen_ids"]
    from collections import Counter
    dupes = sorted(g for g, n in Counter(ids).items() if n > 1)
    unexpected, by_design = classify_collisions(scan["body_nodes"], scan["specs"])
    benign = set(known_benign_items)
    collision_items = sorted({i for g in unexpected for i in g["items"]})
    checks = {
        "generation_ids_unique": {"passed": not dupes and not scan["missing_think"], "ids": len(ids),
                                  "unique": len(set(ids)), "duplicates": dupes[:20],
                                  "parsed_without_think_id": len(scan["missing_think"])},
        "request_body_collisions": {"passed": len(unexpected) == len(benign) and set(collision_items) == benign
                                    and all(len(g["items"]) == 1 and len(g["nodes"]) == 2 for g in unexpected),
                                    "raw_think_bodies": scan["raw_think_bodies"], "by_design_bank_groups": by_design,
                                    "unexpected_groups": unexpected, "known_benign_items": sorted(benign)},
        "observed_model_provider": {"passed": not scan["model_bad"] and not scan["provider_bad"],
                                    "parsed": scan["parsed"], "model_violations": scan["model_bad"][:20],
                                    "provider_violations": scan["provider_bad"][:20]},
        "generation_metadata": check_metadata(legacy_root),
        "forced_budget_check": check_forced_budget(legacy_root),
        "split_overlap": check_splits(splits),
    }
    passed = all(c["passed"] is not False for c in checks.values()) and \
        all(c["passed"] for k, c in checks.items() if k != "split_overlap")
    return {"gate": "e0_gate", "passed": passed, "legacy_root": str(legacy_root), "checks": checks,
            "evaluated_at": _utc()}


def write_e0_gate(study_root: str | Path, result: Mapping[str, Any]) -> Path:
    target = Path(study_root) / "gates" / "e0_gate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(result), indent=1, sort_keys=True, default=str))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_allocation.passk.e0_gate")
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--study-root", required=True, help=".../passk_cont_v1 (writes gates/e0_gate.json)")
    parser.add_argument("--splits", default=None)
    args = parser.parse_args(argv)
    splits = json.loads(Path(args.splits).read_text()) if args.splits else None
    result = e0_gate(args.legacy_root, splits=splits)
    target = write_e0_gate(args.study_root, result)
    print(json.dumps({"e0_gate": str(target), "passed": result["passed"],
                      "checks": {k: v["passed"] for k, v in result["checks"].items()}}, indent=1))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
