"""WP-G test fixtures: a splits.json with the contract section 5.1 schema, built from the legacy panels when the
legacy run root is readable (L96 = legacy main, D32 = legacy dev, legacy order) and from synthetic ids otherwise;
T200 is always synthetic here (the lead's splits.py draws the real one). Tasks are real legacy PublicTask records
where available and synthetic ones for the rest. Nothing is written outside ``tmp_path``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_allocation.contracts import BENCHMARKS, OutputKind, PublicTask, dump_record, load_record

LEGACY_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")
T200_QUOTA = {"aime": 20, "hmmt": 20, "omni_math_hard": 100, "gpqa_diamond": 60}


def legacy_available() -> bool:
    return (LEGACY_ROOT / "manifests" / "panels.json").is_file()


def synthetic_task(item_id: str) -> PublicTask:
    benchmark = item_id.split("/", 1)[0]
    if benchmark == "gpqa_diamond":
        return PublicTask(item_id=item_id, benchmark=benchmark, source_revision="fixture",
                          prompt=f"Question {item_id}\n\nOptions:\nA. a\nB. b\nC. c\nD. d",
                          output_kind=OutputKind.OPTION, allowed_choices=("A", "B", "C", "D"))
    return PublicTask(item_id=item_id, benchmark=benchmark, source_revision="fixture",
                      prompt=f"Compute the answer to problem {item_id}.", output_kind=OutputKind.MATH)


def legacy_tasks() -> dict[str, PublicTask]:
    out = {}
    for benchmark in BENCHMARKS:
        source = LEGACY_ROOT / "manifests" / "public" / f"{benchmark}.jsonl"
        for line in source.read_text().splitlines():
            if line.strip():
                task = load_record(PublicTask, json.loads(line))
                out[task.item_id] = task
    return out


def legacy_panels() -> dict[str, dict[str, list[str]]]:
    data = json.loads((LEGACY_ROOT / "manifests" / "panels.json").read_text())
    return {b: {k: list(v) for k, v in row.items() if k.endswith("_item_ids")} for b, row in data.items()}


def fixture_splits(*, use_legacy: bool | None = None, t200_quota: dict[str, int] | None = None,
                   l96_per: int = 24, d32_per: int = 8, c48_per: int = 12) -> tuple[dict[str, Any], dict[str, PublicTask]]:
    """(splits, tasks). Panels: L96/D32 legacy (or synthetic), C48 = first ``c48_per`` L96 items per benchmark,
    V48 = C48, T200 synthetic per ``t200_quota``, X80 = a prefix of T200."""
    use_legacy = legacy_available() if use_legacy is None else use_legacy
    quota = dict(T200_QUOTA if t200_quota is None else t200_quota)
    tasks: dict[str, PublicTask] = {}
    if use_legacy:
        panels = legacy_panels()
        tasks.update(legacy_tasks())
        l96 = [i for b in BENCHMARKS for i in panels[b]["main_item_ids"][:l96_per]]
        d32 = [i for b in BENCHMARKS for i in panels[b]["dev_item_ids"][:d32_per]]
    else:
        l96 = [f"{b}/main/{k}" for b in BENCHMARKS for k in range(l96_per)]
        d32 = [f"{b}/dev/{k}" for b in BENCHMARKS for k in range(d32_per)]
    c48 = [i for b in BENCHMARKS for i in [x for x in l96 if x.split("/", 1)[0] == b][:c48_per]]
    t200 = [f"{b}/t200/{k}" for b in BENCHMARKS for k in range(quota.get(b, 0))]
    for item in [*l96, *d32, *t200]:
        tasks.setdefault(item, synthetic_task(item))
    x80 = t200[: max(1, len(t200) * 2 // 5)]
    splits = {
        "version": "passk-splits-v1", "passk_seed": 20260925,
        "panels": {"L96": l96, "C48": c48, "V48": list(c48), "T200": t200, "D32": d32, "X80": x80},
        "c48": {"strata": {}, "inclusion_prob": {i: 0.5 for i in c48}},
        "t200_strata": {},
        "assignment_t200": {i: ("mancer" if k % 2 == 0 else "parasail") for k, i in enumerate(t200)},
        "salts": {"wave": "passk_cont_v1/wave|seed=20260925"}, "sha256_inputs": {},
    }
    return splits, tasks


def write_fixture_root(artifacts: Path, splits: dict[str, Any], tasks: dict[str, PublicTask],
                       privates: dict[str, Any] | None = None) -> None:
    """A prepared harness run root: public (and optionally private) manifests for every panel item, panels.json in
    legacy format (dev = D32, main = L96, reserve = T200), waves.json = [] and manifests/splits.json."""
    from agent_allocation.contracts import PanelDraw
    manifests = artifacts / "manifests"
    (manifests / "public").mkdir(parents=True, exist_ok=True)
    panel_items = [*splits["panels"]["L96"], *splits["panels"]["D32"], *splits["panels"]["T200"]]
    for benchmark in BENCHMARKS:
        rows = [dump_record(tasks[i]) for i in panel_items if i.split("/", 1)[0] == benchmark]
        (manifests / "public" / f"{benchmark}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                                                         for r in rows))
    if privates is not None:
        (artifacts / "private").mkdir(parents=True, exist_ok=True)
        for benchmark in BENCHMARKS:
            rows = [dump_record(privates[i]) for i in panel_items if i.split("/", 1)[0] == benchmark and i in privates]
            (artifacts / "private" / f"{benchmark}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                                                                              for r in rows))
    panels = {}
    for benchmark in BENCHMARKS:
        def of(panel: str) -> tuple[str, ...]:
            return tuple(i for i in splits["panels"][panel] if i.split("/", 1)[0] == benchmark)
        panels[benchmark] = dump_record(PanelDraw(benchmark=benchmark, selection_seed=20260917, procedure="fixture",
                                                  dev_item_ids=of("D32"), main_item_ids=of("L96"),
                                                  reserve_item_ids=of("T200")))
    (manifests / "panels.json").write_text(json.dumps(panels, indent=1, sort_keys=True))
    (manifests / "waves.json").write_text("[]")
    (manifests / "splits.json").write_text(json.dumps(splits, indent=1, sort_keys=True))
