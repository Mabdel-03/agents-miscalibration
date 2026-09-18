"""Provider shards: one paid dispatcher per provider endpoint, each on a DISJOINT set of items.

The study config lists ``shards: [{id, endpoint_tag, share}]``. Items are assigned to shards by a seeded,
benchmark-balanced deterministic hash (:func:`shard_assignment`): within every benchmark the items are ordered by
``sha256(seed|benchmark|item_id)`` and cut into per-shard blocks whose sizes follow ``share`` (largest cumulative
deficit first, so the benchmark remainders balance out across shards). Every item of one shard stays in it, so every
within-item contrast is same-provider; the provider is a reported covariate after the merge.

Layout under the run root (``--run-dir``, where ``prepare-data`` wrote the full manifests)::

    <run_root>/shards/assignment.json         {item_id: shard_id}, the per-panel counts, the config sha256
    <run_root>/shards/<shard_id>/             a complete run dir (RunPaths): own locks/, manifests/ (restricted to
                                              the shard's items), private/, state/ (ledger journal, state backup),
                                              raw/, parsed/, grades/, selections/, tables/, reports/, control/
    <run_root>/shards/<shard_id>/state/shard_config.json   the derived config the shard runs under
    <run_root>/merged/                        union of the sealed shard outputs, re-analysed (``merge-shards``)

A shard's derived config (:func:`shard_config`) sets ``provider.endpoint_tag`` to the shard's tag, scales every
budget figure in :data:`config.SHARD_SCALED_BUDGET_KEYS` by ``share/sum(share)`` (rounded down to the cent) and
carries a ``shard`` block that ``validate_config`` checks against the preset ceiling.

CLI verbs: ``shard-plan`` (print, and with ``--write`` materialize, the assignment and per-shard budgets) and
``merge-shards`` (see :mod:`agent_allocation.analysis.merge`). Every live verb (``canary``, ``pilot``, ``freeze``,
``run``, ``seal``, ``calibrate``) takes ``--shard ID`` with ``--run-dir`` naming the run root.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.config import (
    SHARD_SCALED_BUDGET_KEYS, config_hash, load_config, scale_usd, shard_fraction, validate_config,
)
from agent_allocation.paths import RunPaths

SHARDS_DIRNAME = "shards"
MERGED_DIRNAME = "merged"
ASSIGNMENT_NAME = "assignment.json"
SHARD_CONFIG_NAME = "shard_config.json"
ASSIGNMENT_VERSION = "shard-assignment-v1"


class ShardError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------
# Shards and budgets
# ------------------------------------------------------------------------------------------------------------------

def shard_entries(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    shards = cfg.get("shards") or []
    if not shards:
        raise ShardError("config declares no shards")
    return [dict(s) for s in shards]


def shard_ids(cfg: Mapping[str, Any]) -> list[str]:
    return [s["id"] for s in shard_entries(cfg)]


def total_share(cfg: Mapping[str, Any]) -> Decimal:
    return sum((Decimal(str(s["share"])) for s in shard_entries(cfg)), Decimal(0))


def _entry(cfg: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    for entry in shard_entries(cfg):
        if entry["id"] == shard_id:
            return entry
    raise ShardError(f"unknown shard {shard_id!r}; config lists {shard_ids(cfg)}")


def base_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The study config a (possibly derived) shard config came from."""
    if cfg.get("shard") is None:
        return dict(cfg)
    base = cfg["shard"].get("base_config")
    if not isinstance(base, Mapping):
        raise ShardError("derived shard config carries no base_config")
    return copy.deepcopy(dict(base))


def shard_config(cfg: Mapping[str, Any], shard_id: str) -> dict[str, Any]:
    """The derived config one shard runs under (endpoint tag set, budget figures scaled to its share)."""
    if cfg.get("shard") is not None:
        cfg = base_config(cfg)
    entry = _entry(cfg, shard_id)
    total = total_share(cfg)
    fraction = shard_fraction(entry["share"], total)
    out = copy.deepcopy(dict(cfg))
    out["provider"] = {**out["provider"], "endpoint_tag": entry["endpoint_tag"]}
    budget = dict(out["budget"])
    for key in SHARD_SCALED_BUDGET_KEYS:
        if key in budget:
            budget[key] = float(scale_usd(budget[key], fraction))
    out["budget"] = budget
    # Per-provider dispatcher limits (e.g. Parasail 429s above ~40 in flight): ``scheduler_by_shard[<id>]`` overrides.
    overrides = (cfg.get("scheduler_by_shard") or {}).get(shard_id) or {}
    unknown = sorted(set(overrides) - set(out.get("scheduler", {})))
    if unknown:
        raise ValueError(f"scheduler_by_shard[{shard_id!r}] has unknown scheduler keys {unknown}")
    out["scheduler"] = {**out.get("scheduler", {}), **overrides}
    out["shard"] = {"id": shard_id, "endpoint_tag": entry["endpoint_tag"], "share": entry["share"],
                    "total_share": str(total),
                    "base_config_sha256": config_hash(cfg), "base_config": copy.deepcopy(dict(cfg))}
    return out


def shard_budget_table(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-shard budget slices (strings, exact cents) for printing and for the launcher."""
    cfg = base_config(cfg)
    rows = []
    for entry in shard_entries(cfg):
        derived = shard_config(cfg, entry["id"])
        rows.append({"shard_id": entry["id"], "endpoint_tag": entry["endpoint_tag"], "share": entry["share"],
                     "fraction": str(shard_fraction(entry["share"], total_share(cfg)).quantize(Decimal("0.0001"))),
                     **{key: str(Decimal(str(derived["budget"][key])).quantize(Decimal("0.01")))
                        for key in SHARD_SCALED_BUDGET_KEYS if key in derived["budget"]}})
    return rows


# ------------------------------------------------------------------------------------------------------------------
# Assignment
# ------------------------------------------------------------------------------------------------------------------

def _benchmark_of(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        return item, item.split("/", 1)[0]
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]), str(item[1])
    return str(item.item_id), str(item.benchmark)


def _hash_key(seed: Any, benchmark: str, item_id: str) -> str:
    return sha256(f"{ASSIGNMENT_VERSION}|seed={seed}|{benchmark}|{item_id}".encode()).hexdigest()


def shard_assignment(cfg: Mapping[str, Any], items: Iterable[Any]) -> dict[str, str]:
    """``{item_id: shard_id}``: seeded, benchmark-balanced and deterministic (independent of input order).

    ``items`` are item ids (benchmark = the id's prefix), ``(item_id, benchmark)`` pairs, or records with
    ``item_id``/``benchmark`` (``PublicTask``). Each benchmark's items are split as evenly as the shares allow; the
    per-benchmark remainders go to the shards furthest below their running proportional target, so shard totals are
    balanced too. No model outcome enters the key."""
    cfg = base_config(cfg)
    entries = shard_entries(cfg)
    weights = [Decimal(str(e["share"])) for e in entries]
    total = sum(weights, Decimal(0))
    by_bench: dict[str, list[str]] = {}
    for item in items:
        item_id, benchmark = _benchmark_of(item)
        by_bench.setdefault(benchmark, [])
        if item_id not in by_bench[benchmark]:
            by_bench[benchmark].append(item_id)
    seed = cfg.get("seed")
    assigned_total = [0] * len(entries)
    seen_total = 0
    out: dict[str, str] = {}
    for benchmark in sorted(by_bench):
        ids = sorted(by_bench[benchmark], key=lambda i: (_hash_key(seed, benchmark, i), i))
        n = len(ids)
        exact = [w * n / total for w in weights]
        quota = [int(x) for x in exact]
        seen_total += n
        leftover = n - sum(quota)
        # Remainders: the shard furthest below its cumulative proportional target first; ties by shard order.
        order = sorted(range(len(entries)),
                       key=lambda s: (-(weights[s] * seen_total / total - assigned_total[s] - quota[s]), s))
        for s in order[:leftover]:
            quota[s] += 1
        start = 0
        for s, count in enumerate(quota):
            for item_id in ids[start:start + count]:
                out[item_id] = entries[s]["id"]
            start += count
            assigned_total[s] += count
    return out


def items_of(assignment: Mapping[str, str], shard_id: str) -> set[str]:
    return {item for item, shard in assignment.items() if shard == shard_id}


# ------------------------------------------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------------------------------------------

def run_root(run_dir: str | Path | None) -> Path:
    """The run root exactly as ``RunPaths.for_run`` would resolve it (without touching node-local state)."""
    from agent_allocation.contracts import STUDY_ID
    from agent_allocation.paths import ARTIFACTS, PROJECT_ROOT
    chosen = Path(run_dir or os.environ.get("ALLOCATION_RUN_DIR") or (ARTIFACTS / STUDY_ID))
    return chosen if chosen.is_absolute() else PROJECT_ROOT / chosen


def shard_run_dir(root: str | Path, shard_id: str) -> Path:
    return Path(root) / SHARDS_DIRNAME / shard_id


def merged_run_dir(root: str | Path) -> Path:
    return Path(root) / MERGED_DIRNAME


def assignment_path(root: str | Path) -> Path:
    return Path(root) / SHARDS_DIRNAME / ASSIGNMENT_NAME


def shard_paths(root: str | Path, shard_id: str, *, local_state_dir: str | Path | None = None) -> RunPaths:
    return RunPaths.for_run(shard_run_dir(run_root(root), shard_id), local_state_dir=local_state_dir)


def resolve_shard(cfg: Mapping[str, Any], run_dir: str | Path | None, shard_id: str | None
                  ) -> tuple[dict[str, Any], RunPaths]:
    """``(config, paths)`` for a live verb. Without ``--shard`` a config that declares shards is refused (an
    unsharded dispatcher would run with no endpoint tag); with it, the derived config and the shard's run dir."""
    if shard_id is None:
        if cfg.get("shards") and cfg.get("shard") is None:
            raise ShardError("this config declares shards: pass --shard ID (with --run-dir naming the run root)")
        return dict(cfg), RunPaths.for_run(run_dir)
    derived = shard_config(cfg, shard_id)
    root = run_root(run_dir)
    return derived, RunPaths.for_run(shard_run_dir(root, shard_id))


def shard_filter(cfg: Mapping[str, Any], paths: RunPaths) -> set[str] | None:
    """Items a shard config may build graphs for (``None`` for an unsharded config). Read from the shard dir's own
    copy of the assignment, falling back to the run root's."""
    shard = cfg.get("shard")
    if not shard:
        return None
    for candidate in (paths.manifests_dir / "shard_assignment.json", assignment_path(paths.run_dir.parent.parent)):
        if candidate.is_file():
            data = json.loads(candidate.read_text())
            return items_of(data["assignment"], shard["id"])
    raise ShardError(f"no shard assignment for shard {shard['id']!r} (run `allocation shard-plan --write` first)")


def filter_graph_waves(waves: Sequence[Sequence[Any]], allowed: set[str] | None) -> list[list[Any]]:
    """Restrict item-graph waves to a shard's items, dropping waves left empty."""
    if allowed is None:
        return [list(w) for w in waves]
    out = [[g for g in wave if g.item_id in allowed] for wave in waves]
    return [w for w in out if w]


# ------------------------------------------------------------------------------------------------------------------
# Materialization
# ------------------------------------------------------------------------------------------------------------------

def root_assignment(cfg: Mapping[str, Any], root_paths: RunPaths) -> dict[str, Any]:
    """Assignment over every panel item of the prepared run root, each panel (dev, main, reserve) balanced alone."""
    from agent_allocation.manifest import load_panels
    panels = load_panels(root_paths)
    assignment: dict[str, str] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for panel in ("dev_item_ids", "main_item_ids", "reserve_item_ids"):
        pairs = [(i, b) for b, draw in panels.items() for i in getattr(draw, panel)]
        part = shard_assignment(cfg, pairs)
        assignment.update(part)
        by_bench = dict(pairs)
        table: dict[str, dict[str, int]] = {}
        for item, shard in part.items():
            table.setdefault(shard, {}).setdefault(by_bench[item], 0)
            table[shard][by_bench[item]] += 1
        counts[panel.replace("_item_ids", "")] = {s: dict(sorted(v.items())) for s, v in sorted(table.items())}
    return {"version": ASSIGNMENT_VERSION, "seed": cfg.get("seed"), "config_sha256": config_hash(base_config(cfg)),
            "shards": shard_entries(cfg), "counts": counts, "assignment": dict(sorted(assignment.items()))}


def _filter_jsonl(source: Path, target: Path, keep: set[str], *, mode: int | None = None) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("item_id") in keep:
            kept.append(line + "\n")
    target.write_text("".join(kept), encoding="utf-8")
    if mode is not None:
        os.chmod(target, mode)
    return len(kept)


def materialize_shard(cfg: Mapping[str, Any], root_paths: RunPaths, shard_id: str, assignment: Mapping[str, Any],
                      *, local_state_dir: str | Path | None = None) -> dict[str, Any]:
    """Write one shard's run dir: manifests restricted to its items, private records, the dataset lock, the derived
    config and a copy of the assignment. Idempotent; refuses to rewrite manifests once the shard has collected."""
    from agent_allocation.contracts import Wave, dump_record
    from agent_allocation.manifest import load_panels, load_waves, write_atomic
    paths = RunPaths.for_run(shard_run_dir(root_paths.run_dir, shard_id), local_state_dir=local_state_dir)
    keep = items_of(assignment["assignment"], shard_id)
    if paths.ledger_journal.exists() and paths.waves.exists():
        existing = {i for w in json.loads(paths.waves.read_text()) for i in w["item_ids"]}
        wanted = {i for w in load_waves(root_paths) for i in w.item_ids if i in keep}
        if existing != wanted:
            raise ShardError(f"shard {shard_id} has collected under a different assignment; refusing to rewrite")
    paths.ensure()
    public_dir = root_paths.manifests_dir / "public"
    counts = {}
    for source in sorted(public_dir.glob("*.jsonl")):
        counts[source.stem] = _filter_jsonl(source, paths.manifests_dir / "public" / source.name, keep)
    if root_paths.private_dir.is_dir():
        for source in sorted(root_paths.private_dir.glob("*.jsonl")):
            _filter_jsonl(source, paths.private_dir / source.name, keep, mode=0o600)
        for sub in sorted(p for p in root_paths.private_dir.iterdir() if p.is_dir()):
            shutil.copytree(sub, paths.private_dir / sub.name, dirs_exist_ok=True)
    panels = {b: replace(d, dev_item_ids=tuple(i for i in d.dev_item_ids if i in keep),
                         main_item_ids=tuple(i for i in d.main_item_ids if i in keep),
                         reserve_item_ids=tuple(i for i in d.reserve_item_ids if i in keep))
              for b, d in load_panels(root_paths).items()}
    write_atomic(paths.panels, json.dumps({b: dump_record(d) for b, d in panels.items()}, indent=1, sort_keys=True))
    waves = [Wave(index=w.index, item_ids=tuple(i for i in w.item_ids if i in keep), panels=w.panels)
             for w in load_waves(root_paths)]
    waves = [w for w in waves if w.item_ids]
    write_atomic(paths.waves, json.dumps([dump_record(w) for w in waves], indent=1, sort_keys=True))
    for name in ("exclusions.json", "alias_report.json"):
        if (root_paths.manifests_dir / name).is_file():
            shutil.copy2(root_paths.manifests_dir / name, paths.manifests_dir / name)
    if root_paths.dataset_lock.is_file() and not paths.dataset_lock.exists():
        shutil.copy2(root_paths.dataset_lock, paths.dataset_lock)
    root_digests = {str(p.relative_to(root_paths.run_dir)): sha256(p.read_bytes()).hexdigest()
                    for p in sorted([*public_dir.glob("*.jsonl"), root_paths.panels, root_paths.waves]) if p.is_file()}
    write_atomic(paths.manifests_dir / "shard_assignment.json",
                 json.dumps({**assignment, "shard_id": shard_id, "root_manifest_sha256": root_digests},
                            indent=1, sort_keys=True))
    derived = shard_config(cfg, shard_id)
    write_atomic(paths.state_dir / SHARD_CONFIG_NAME, json.dumps(derived, indent=1, sort_keys=True))
    return {"shard_id": shard_id, "run_dir": str(paths.run_dir), "items": len(keep), "public_rows": counts,
            "waves": len(waves), "problems": validate_config(derived)}


# ------------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------------

def cmd_shard_plan(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation shard-plan",
                                     description="Print (and with --write materialize) the provider shard plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None, help="run root holding the prepared manifests")
    parser.add_argument("--write", action="store_true", help="write shards/assignment.json and every shard run dir")
    parser.add_argument("--show-items", action="store_true", help="print the full item -> shard map")
    args = parser.parse_args(argv)
    cfg = base_config(load_config(args.config))
    problems = validate_config(cfg)
    budgets = shard_budget_table(cfg)
    out: dict[str, Any] = {"shards": budgets, "config_problems": problems}
    root = run_root(args.run_dir)
    root_paths = RunPaths.for_run(root, local_state_dir=os.environ.get("ALLOCATION_LOCAL_STATE") or root / "state")
    if root_paths.panels.is_file():
        assignment = root_assignment(cfg, root_paths)
        out["counts"] = assignment["counts"]
        if args.show_items:
            out["assignment"] = assignment["assignment"]
        if args.write:
            if problems:
                print(json.dumps(out, indent=1, sort_keys=True))
                return 1
            target = assignment_path(root_paths.run_dir)
            if target.is_file():
                previous = json.loads(target.read_text())
                if previous.get("assignment") != assignment["assignment"]:
                    raise ShardError(f"{target} exists with a different assignment; refusing to overwrite")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(assignment, indent=1, sort_keys=True))
            out["written"] = [materialize_shard(cfg, root_paths, sid, assignment) for sid in shard_ids(cfg)]
    else:
        out["counts"] = None
        out["note"] = f"no prepared manifests under {root_paths.run_dir} (run prepare-data first to see the assignment)"
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 1 if problems else 0
