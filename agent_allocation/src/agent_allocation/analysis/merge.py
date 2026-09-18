"""``allocation merge-shards``: one analysis over the union of the provider shards' sealed outputs.

Seam: every table ``analyze`` writes is a pure function of a :class:`~agent_allocation.analysis.load.RunData`
(manifests, parsed results, grades, sealed selections, ledger). Each shard run dir is loaded on its own -- so every
shard's selection seals are verified against its own files, never rewritten -- and the loaded records are unioned
into one ``RunData`` rooted at ``<run_root>/merged``; the unchanged ``build_analysis``/``write_analysis`` then
recompute every summary, contrast, interval and cost table over the union. Items are disjoint across shards (checked:
an item id in two shards refuses the merge), so the union is exactly the single-run input the same items would give.

Union rules:
* tasks, panels, waves, jobs, policies, parsed results, public checks, grades, node states, render reports: dict
  union (a key in two shards refuses); waves with the same index are joined item-wise.
* selections: each shard's batches keep their verification result and are renumbered with a per-shard offset
  (``selection_batch`` follows), so ``RunData.is_sealed`` means "sealed and verified in its own shard".
* ledger: shard journals concatenated in shard order with ``seq`` renumbered (the replay invariant is then checked
  against the base study ceiling); provider ``/key`` snapshots are unioned in time order (shards bill one key).
* raw pointers: ``GenerationResult.raw_response_path`` is re-expressed relative to ``merged/`` so replay and
  effective-depth readers find the shard's raw file.
* covariates: every written table with an ``item_id`` column gains ``shard_id`` and ``endpoint_tag``.

Outputs under ``<run_root>/merged/``: ``tables/``, ``reports/``, ``figures/`` exactly as ``analyze`` writes them,
plus ``merge_manifest.json`` (shards, their input digests, item counts, endpoint locks).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from agent_allocation.analysis.load import PartitionSeal, RunData, load_run, resolve_config
from agent_allocation.contracts import Wave
from agent_allocation.paths import RunPaths

COVARIATE_COLUMNS = ("shard_id", "endpoint_tag")


class MergeError(RuntimeError):
    pass


def _shard_run_paths(run_dir: Path) -> RunPaths:
    return RunPaths.for_run(run_dir, local_state_dir=run_dir / "state")


def discover_shards(root: Path, shard_ids: Sequence[str] | None = None) -> list[tuple[str, Path]]:
    from agent_allocation.shards import SHARDS_DIRNAME
    base = Path(root) / SHARDS_DIRNAME
    if shard_ids:
        found = [(sid, base / sid) for sid in shard_ids]
        missing = [str(p) for _, p in found if not p.is_dir()]
        if missing:
            raise MergeError(f"shard run dirs missing: {missing}")
        return found
    if not base.is_dir():
        raise MergeError(f"no shard run dirs under {base}")
    return [(p.name, p) for p in sorted(base.iterdir()) if p.is_dir()]


def _union(target: dict, source: Mapping, what: str, shard: str) -> None:
    clash = sorted(set(target) & set(source))
    if clash:
        raise MergeError(f"{what}: {len(clash)} keys appear in more than one shard (e.g. {clash[0]} again in {shard}); "
                         "shards must hold disjoint items")
    target.update(source)


def merge_run_data(runs: Sequence[tuple[str, RunData]], merged_paths: RunPaths, config: Mapping[str, Any],
                   *, endpoint_tags: Mapping[str, str | None] | None = None) -> tuple[RunData, dict[str, str]]:
    """Union of shard ``RunData`` (see module docstring). Returns the merged run and ``{item_id: shard_id}``."""
    merged = RunData(paths=merged_paths, config=dict(config),
                     config_source=f"base study config; merged from shards {[s for s, _ in runs]}")
    item_shard: dict[str, str] = {}
    waves: dict[int, Wave] = {}
    batch_offset: dict[str, int] = {}
    seq = 0
    for shard, run in runs:
        items = {spec.item_id for specs in run.jobs.values() for spec in specs.values()}
        clash = sorted(i for i in items if i in item_shard)
        if clash:
            raise MergeError(f"item {clash[0]} is in shards {item_shard[clash[0]]} and {shard}; items must be disjoint")
        item_shard.update({i: shard for i in items})
        for item_id, task in run.tasks.items():
            if item_id in items and item_id in merged.tasks and item_shard.get(item_id) != shard:
                raise MergeError(f"public task {item_id} is planned in more than one shard")
            merged.tasks.setdefault(item_id, task)
        for bench, draw in run.panels.items():
            if bench in merged.panels:
                prev = merged.panels[bench]
                merged.panels[bench] = replace(prev, dev_item_ids=prev.dev_item_ids + draw.dev_item_ids,
                                               main_item_ids=prev.main_item_ids + draw.main_item_ids,
                                               reserve_item_ids=prev.reserve_item_ids + draw.reserve_item_ids)
            else:
                merged.panels[bench] = draw
        for wave in run.waves:
            if wave.index in waves:
                prev = waves[wave.index]
                if tuple(prev.panels) != tuple(wave.panels):
                    raise MergeError(f"wave {wave.index} has panels {prev.panels} in one shard, {wave.panels} in {shard}")
                waves[wave.index] = replace(prev, item_ids=prev.item_ids + wave.item_ids)
            else:
                waves[wave.index] = wave
        for partition, specs in run.jobs.items():
            _union(merged.jobs.setdefault(partition, {}), specs, f"jobs/{partition}", shard)
        for partition, cps in run.policies.items():
            merged.policies.setdefault(partition, []).extend(cps)
        parsed = {}
        for node_id, (result, candidate) in run.parsed.items():
            if result.raw_response_path:
                rel = os.path.relpath(run.paths.run_dir / result.raw_response_path, merged_paths.run_dir)
                result = replace(result, raw_response_path=rel)
            parsed[node_id] = (result, candidate)
        _union(merged.parsed, parsed, "parsed results", shard)
        _union(merged.public_checks, run.public_checks, "public checks", shard)
        _union(merged.grades, run.grades, "grades", shard)
        _union(merged.node_states, run.node_states, "node states", shard)
        _union(merged.render_reports, run.render_reports, "render reports", shard)
        for partition, seal in run.seals.items():
            offset = batch_offset.get(partition, 0)
            batches = tuple(replace(b, batch=b.batch + offset) for b in seal.batches)
            prev = merged.seals.get(partition)
            problems = tuple(f"shard {shard}: {p}" for p in seal.problems)
            if prev is None:
                merged.seals[partition] = PartitionSeal(partition, batches, problems)
            else:
                merged.seals[partition] = PartitionSeal(partition, prev.batches + batches, prev.problems + problems)
            _union(merged.selections.setdefault(partition, {}), run.selections.get(partition, {}),
                   f"selections/{partition}", shard)
            merged.selection_batch.setdefault(partition, {}).update(
                {pid: b + offset for pid, b in run.selection_batch.get(partition, {}).items()})
            numbers = [b.batch for b in seal.batches]
            batch_offset[partition] = offset + (max(numbers) + 1 if numbers else 0)
        for event in run.ledger:
            seq += 1
            merged.ledger.append(replace(event, seq=seq))
        merged.gates.extend(run.gates)
        merged.amendments.extend({**a, "shard_id": shard} for a in run.amendments)
        merged.provider_snapshots.extend({**s, "shard_id": shard} for s in run.provider_snapshots)
        merged.context_limit_events.extend(run.context_limit_events)
        merged.superseded_grades += run.superseded_grades
        merged.superseded_public_checks += run.superseded_public_checks
        merged.problems.extend(f"shard {shard}: {p}" for p in run.problems)
        merged.input_files.extend(run.input_files)
    merged.waves = [waves[k] for k in sorted(waves)]
    merged.provider_snapshots.sort(key=lambda r: (str(r.get("fetched_at", "")), str(r.get("endpoint", ""))))
    digest = sha256()
    for shard, run in runs:
        digest.update(f"{shard}:{run.input_digest()}\n".encode())
    merged._input_digest = digest.hexdigest()           # shard files live outside merged/; digest of shard digests
    return merged, item_shard


def add_covariates(frame: pd.DataFrame, item_shard: Mapping[str, str],
                   endpoint_tags: Mapping[str, str | None]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "item_id" not in frame.columns:
        return frame
    out = frame.copy()
    out["shard_id"] = out["item_id"].map(lambda i: item_shard.get(str(i)))
    out["endpoint_tag"] = out["shard_id"].map(lambda s: endpoint_tags.get(s) if s is not None else None)
    return out


def merge_shards(root: str | Path, *, config: Mapping[str, Any] | None = None, shard_ids: Sequence[str] | None = None,
                 replicates: int | None = None, figures: bool = True, out_dir: str | Path | None = None):
    """Load every shard run dir under ``root``, union them, analyse, and write to ``<root>/merged``."""
    from agent_allocation.reporting.cli import build_analysis, write_analysis
    from agent_allocation.shards import MERGED_DIRNAME, base_config
    root = Path(root)
    shards = discover_shards(root, shard_ids)
    runs: list[tuple[str, RunData]] = []
    tags: dict[str, str | None] = {}
    base: dict[str, Any] | None = dict(config) if config is not None else None
    locks: dict[str, Any] = {}
    for shard, run_dir in shards:
        paths = _shard_run_paths(run_dir)
        shard_cfg, source = resolve_config(paths, None)
        shard_cfg_file = paths.state_dir / "shard_config.json"
        if "shard" not in shard_cfg and shard_cfg_file.is_file():
            shard_cfg, source = json.loads(shard_cfg_file.read_text()), str(shard_cfg_file)
        run = load_run(paths, shard_cfg, config_source=source)
        runs.append((shard, run))
        tags[shard] = (shard_cfg.get("shard") or {}).get("endpoint_tag") or shard_cfg.get("provider", {}).get(
            "endpoint_tag")
        if base is None and shard_cfg.get("shard"):
            base = base_config(shard_cfg)
        if paths.endpoint_lock.is_file():
            locks[shard] = json.loads(paths.endpoint_lock.read_text())
    if base is None:
        raise MergeError("no base config: pass --config or run shards under derived shard configs")
    target = Path(out_dir) if out_dir else root / MERGED_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    merged_paths = _shard_run_paths(target)
    for directory in (merged_paths.tables_dir, merged_paths.figures_dir, merged_paths.reports_dir,
                      merged_paths.locks_dir):
        directory.mkdir(parents=True, exist_ok=True)
    merged, item_shard = merge_run_data(runs, merged_paths, base, endpoint_tags=tags)
    analysis = build_analysis(merged, replicates=replicates)
    analysis.candidates = add_covariates(analysis.candidates, item_shard, tags)
    analysis.outcomes = add_covariates(analysis.outcomes, item_shard, tags)
    for key, value in list(analysis.extras.items()):
        if isinstance(value, pd.DataFrame):
            analysis.extras[key] = add_covariates(value, item_shard, tags)
    metrics = analysis.extras.get("item_metrics")
    if isinstance(metrics, dict):
        for key, value in list(metrics.items()):
            metrics[key] = add_covariates(value, item_shard, tags)
    write_analysis(analysis, figures=figures)
    manifest = {
        "root": str(root), "merged_dir": str(target),
        "shards": [{"shard_id": s, "run_dir": str(r.paths.run_dir), "endpoint_tag": tags.get(s),
                    "input_digest": r.input_digest(), "items": sum(1 for v in item_shard.values() if v == s),
                    "nodes": sum(len(j) for j in r.jobs.values()),
                    "sealed_partitions": sorted(p for p, seal in r.seals.items() if seal.valid)}
                   for s, r in runs],
        "merged_input_digest": merged.input_digest(), "endpoint_locks": locks,
        "items": len(item_shard), "final": analysis.final,
    }
    (target / "merge_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
    return analysis, manifest


def cmd_merge_shards(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation merge-shards",
                                     description="Union the sealed shard outputs and analyse them as one run.")
    parser.add_argument("--run-dir", required=True, help="run root holding shards/<id>/")
    parser.add_argument("--config", default=None, help="base study config (default: from the shards' config locks)")
    parser.add_argument("--shard", action="append", default=None, help="restrict to these shard ids (repeatable)")
    parser.add_argument("--out", default=None, help="output dir (default <run-dir>/merged)")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    from agent_allocation.config import load_config
    from agent_allocation.shards import base_config, run_root
    cfg = base_config(load_config(args.config)) if args.config else None
    analysis, manifest = merge_shards(run_root(args.run_dir), config=cfg, shard_ids=args.shard,
                                      replicates=args.replicates, figures=not args.no_figures, out_dir=args.out)
    state = "final" if analysis.final else "INTERIM (partial or unreconciled)"
    print(json.dumps({k: manifest[k] for k in ("merged_dir", "items", "final")}, indent=1), file=sys.stdout)
    print(f"merge-shards: {state}; reports in {analysis.run.paths.reports_dir}", file=sys.stderr)
    return 0
