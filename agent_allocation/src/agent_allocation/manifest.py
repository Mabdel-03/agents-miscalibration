"""``allocation prepare-data`` (lead-owned): inventories -> seeded panels -> balanced waves -> manifests + dataset lock.

Wave construction (plan design decision 7): every wave holds four items per benchmark in ``BENCHMARKS`` order
(one per BBH subtask), and nested panels are wave prefixes so a budget stop trims only the tail of main:
extension = the first ``extension_items/16`` waves, high = the first ``high_items/16``, replication = the first
``replication_items/16``. The pilot uses each benchmark's disjoint development items.

The dataset lock is write-once. A rerun that would change it refuses (source-version change is a stop rule).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.config import config_hash, load_config, validate_config
from agent_allocation.contracts import (
    BBH_SUBTASKS, BENCHMARKS, WAVE_ITEMS_PER_BENCHMARK, InventoryItem, PanelDraw, Wave, canonical_hash,
    dump_record, load_record,
)
from agent_allocation.paths import RunPaths

ADAPTERS = {
    "math500": "agent_allocation.benchmarks.math500",
    "mmlu_pro": "agent_allocation.benchmarks.mmlu_pro",
    "bbh_logic4": "agent_allocation.benchmarks.bbh_logic4",
    "mbpp_plus": "agent_allocation.benchmarks.mbpp_plus",
}
TEXT_MANIFEST = "agent_allocation.benchmarks.text_manifest"
DEV_PER_BENCHMARK = 8
RESERVE_PER_BENCHMARK = 8


class ManifestRefused(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def write_atomic(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def write_jsonl(path: Path, records: Sequence[Any], *, mode: int | None = None) -> str:
    text = "".join(json.dumps(dump_record(r), sort_keys=True, ensure_ascii=False) + "\n" for r in records)
    write_atomic(path, text, mode=mode)
    return sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path, cls: type) -> list[Any]:
    with open(path, encoding="utf-8") as handle:
        return [load_record(cls, json.loads(line)) for line in handle if line.strip()]


def panel_sizes(cfg: Mapping[str, Any]) -> dict[str, int]:
    panels = cfg["panels"]
    per = {name: panels[name] // len(BENCHMARKS) for name in ("main_items", "high_items", "extension_items",
                                                               "replication_items")}
    per["pilot_items"] = panels["pilot_items"] // len(BENCHMARKS)
    return per


def draw_panels(cfg: Mapping[str, Any], paths: RunPaths) -> tuple[dict[str, list[InventoryItem]], dict[str, PanelDraw]]:
    per = panel_sizes(cfg)
    if per["pilot_items"] != DEV_PER_BENCHMARK:
        raise ManifestRefused("pilot panel must hold 8 development items per benchmark")
    inventories: dict[str, list[InventoryItem]] = {}
    draws: dict[str, PanelDraw] = {}
    seed = int(cfg["seed"])
    for benchmark in BENCHMARKS:
        adapter = importlib.import_module(ADAPTERS[benchmark])
        inventory = list(adapter.load_inventory(cfg, paths))
        inventories[benchmark] = inventory
        if benchmark == "mbpp_plus":
            draw = adapter.select_panels(inventory, seed=seed, n_dev=DEV_PER_BENCHMARK, n_main=per["main_items"],
                                         n_reserve=RESERVE_PER_BENCHMARK)
        else:
            text = importlib.import_module(TEXT_MANIFEST)
            draw = text.select_panels(inventory, benchmark=benchmark, seed=seed, n_dev=DEV_PER_BENCHMARK,
                                      n_main=per["main_items"], n_reserve=RESERVE_PER_BENCHMARK)
        check_draw(benchmark, draw, inventory, n_main=per["main_items"])
        draws[benchmark] = draw
    return inventories, draws


def check_draw(benchmark: str, draw: PanelDraw, inventory: Sequence[InventoryItem], *, n_main: int) -> None:
    by_id = {item.public.item_id: item for item in inventory}
    if draw.benchmark != benchmark:
        raise ManifestRefused(f"{benchmark}: draw labelled {draw.benchmark}")
    groups = {"dev": draw.dev_item_ids, "main": draw.main_item_ids, "reserve": draw.reserve_item_ids}
    seen: set[str] = set()
    for name, ids in groups.items():
        if len(set(ids)) != len(ids):
            raise ManifestRefused(f"{benchmark}: duplicate ids inside {name}")
        if seen & set(ids):
            raise ManifestRefused(f"{benchmark}: {name} overlaps another panel")
        seen |= set(ids)
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ManifestRefused(f"{benchmark}: {name} ids not in inventory: {missing[:3]}")
        ineligible = [i for i in ids if not by_id[i].eligible]
        if ineligible:
            raise ManifestRefused(f"{benchmark}: {name} contains ineligible items {ineligible[:3]}")
    if len(draw.main_item_ids) != n_main or len(draw.dev_item_ids) != DEV_PER_BENCHMARK:
        raise ManifestRefused(f"{benchmark}: wrong panel sizes")
    if benchmark == "bbh_logic4":
        for start in range(0, n_main, WAVE_ITEMS_PER_BENCHMARK):
            block = [by_id[i].public.subtask for i in draw.main_item_ids[start:start + WAVE_ITEMS_PER_BENCHMARK]]
            if tuple(block) != BBH_SUBTASKS:
                raise ManifestRefused(f"bbh_logic4: main block at {start} is not one item per subtask in order")
        dev_subtasks = sorted(by_id[i].public.subtask for i in draw.dev_item_ids)
        if dev_subtasks != sorted(BBH_SUBTASKS * 2):
            raise ManifestRefused("bbh_logic4: development panel must hold two items per subtask")


def build_waves(cfg: Mapping[str, Any], draws: Mapping[str, PanelDraw]) -> list[Wave]:
    panels = cfg["panels"]
    wave_size = int(cfg["scheduler"]["wave_items"])
    if wave_size != WAVE_ITEMS_PER_BENCHMARK * len(BENCHMARKS):
        raise ManifestRefused("wave size must be four items per benchmark")
    n_waves = panels["main_items"] // wave_size
    prefix = {name: panels[f"{name}_items"] // wave_size for name in ("extension", "high", "replication")}
    waves: list[Wave] = []
    for w in range(n_waves):
        items: list[str] = []
        for benchmark in BENCHMARKS:
            block = draws[benchmark].main_item_ids[w * WAVE_ITEMS_PER_BENCHMARK:(w + 1) * WAVE_ITEMS_PER_BENCHMARK]
            items.extend(block)
        memberships = ["main"] + [name for name in ("extension", "high", "replication") if w < prefix[name]]
        waves.append(Wave(index=w, item_ids=tuple(items), panels=tuple(memberships)))
    return waves


def dataset_lock_payload(cfg: Mapping[str, Any], inventories: Mapping[str, Sequence[InventoryItem]],
                         draws: Mapping[str, PanelDraw], manifest_hashes: Mapping[str, str]) -> dict[str, Any]:
    sources = {}
    for benchmark in BENCHMARKS:
        inventory = inventories[benchmark]
        revisions = sorted({item.public.source_revision for item in inventory})
        adapter = importlib.import_module(ADAPTERS[benchmark])
        sources[benchmark] = {
            "source_revisions": revisions,
            "inventory_size": len(inventory),
            "eligible": sum(1 for item in inventory if item.eligible),
            "inventory_hash": canonical_hash([[i.public.item_id, i.private.private_hash, i.eligible]
                                              for i in inventory]),
            "license": getattr(adapter, "LICENSE", None),
            "preprocessing_version": getattr(adapter, "PREPROCESSING_VERSION", None),
            "provenance": getattr(adapter, "SOURCE_PROVENANCE", None),
            "panel_draw": dump_record(draws[benchmark]),
        }
    return {"study_id": cfg["study_id"], "seed": cfg["seed"], "sources": sources, "manifests": dict(manifest_hashes)}


def write_once_json(path: Path, payload: Mapping[str, Any], *, semantic_keys_ignore: Sequence[str] = ("written_at",)) -> str:
    """Write a lock once. An identical rerun is a no-op; a different payload refuses."""
    body = {k: v for k, v in payload.items() if k not in semantic_keys_ignore}
    digest = canonical_hash(body)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("lock_sha256") != digest:
            raise ManifestRefused(f"{path} exists with different content; source or manifest changed after lock")
        return digest
    write_atomic(path, json.dumps({**payload, "lock_sha256": digest, "written_at": utcnow()}, indent=1, sort_keys=True))
    return digest


def prepare_data(cfg: Mapping[str, Any], paths: RunPaths) -> dict[str, Any]:
    problems = validate_config(cfg)
    if problems:
        raise ManifestRefused(f"config invalid: {problems}")
    paths.ensure()
    inventories, draws = draw_panels(cfg, paths)
    waves = build_waves(cfg, draws)
    manifest_hashes: dict[str, str] = {}
    exclusions = {}
    for benchmark in BENCHMARKS:
        by_id = {item.public.item_id: item for item in inventories[benchmark]}
        chosen = [*draws[benchmark].dev_item_ids, *draws[benchmark].main_item_ids, *draws[benchmark].reserve_item_ids]
        manifest_hashes[f"public/{benchmark}"] = write_jsonl(paths.public_manifest(benchmark),
                                                             [by_id[i].public for i in chosen])
        manifest_hashes[f"private/{benchmark}"] = write_jsonl(paths.private_manifest(benchmark),
                                                              [by_id[i].private for i in chosen], mode=0o600)
        exclusions[benchmark] = [{"item_id": i.public.item_id, "reason": i.exclusion_reason}
                                 for i in inventories[benchmark] if not i.eligible]
    panels_text = json.dumps({b: dump_record(d) for b, d in draws.items()}, indent=1, sort_keys=True)
    write_atomic(paths.panels, panels_text)
    manifest_hashes["panels"] = sha256(panels_text.encode()).hexdigest()
    waves_text = json.dumps([dump_record(w) for w in waves], indent=1, sort_keys=True)
    write_atomic(paths.waves, waves_text)
    manifest_hashes["waves"] = sha256(waves_text.encode()).hexdigest()
    write_atomic(paths.exclusions, json.dumps(exclusions, indent=1, sort_keys=True))
    lock_digest = write_once_json(paths.dataset_lock,
                                  {**dataset_lock_payload(cfg, inventories, draws, manifest_hashes),
                                   "config_sha256": config_hash(cfg)})
    return {"dataset_lock_sha256": lock_digest, "waves": len(waves), "manifests": manifest_hashes,
            "exclusions": {b: len(v) for b, v in exclusions.items()}}


def load_waves(paths: RunPaths) -> list[Wave]:
    return [load_record(Wave, row) for row in json.loads(paths.waves.read_text())]


def load_panels(paths: RunPaths) -> dict[str, PanelDraw]:
    return {b: load_record(PanelDraw, row) for b, row in json.loads(paths.panels.read_text()).items()}


def cmd_prepare_data(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="allocation prepare-data")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    paths = RunPaths.for_run(args.run_dir)
    summary = prepare_data(cfg, paths)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0
