"""passk_cont_v1 node banks (WP-G): the frozen node list of every partition of contract section 3.

Every node is a standalone forced generation built with :func:`live.standalone_spec` (kind ROOT at ``A``, DEEP at
``>= 2A``; ``root_slot = draw_id = replicate``) and wrapped per (partition, item) by :func:`live.nodes_only_graph`.
Within an item the nodes are emitted in descending budget (then ascending replicate), so with DEEP and ROOT slot 0 in
critical class 0 and FIFO within a class, the dispatcher sends 16A -> 8A -> 4A -> 2A -> A.

| partition   | panel | cells (multiplier of A: replicates)             | per item | items | nodes |
|-------------|-------|-------------------------------------------------|---------:|------:|------:|
| smoke       | D32*  | 1: r100                                         |        1 |    12 |    12 |
| e1_fill     | L96   | 8: r1; 4: r1-3; 2: r1-7                         |       11 |    96 |  1056 |
| c48_extra   | C48   | 16: r1-7; 8: r2-7; 4: r4-7                      |       17 |    48 |   816 |
| v48_bank    | V48   | 16: r0; 8: r0-1; 4: r0-3; 2: r0-7; 1: r0-15     |       31 |    48 |  1488 |
| t200_bank   | T200  | same 31-node bank                               |       31 |   200 |  6200 |
| t200_diag   | T200  | 1: r0-4                                         |        5 |   200 |  1000 |

(*) smoke = the first 3 items of each benchmark's D32 list, in D32 order.

Waves: stage 1 = ``c48_extra``, ``e1_fill``, ``t200_diag`` (in that order, chunked by ``scheduler.wave_items``);
stage 2 = ``v48_bank`` and ``t200_bank`` interleaved in seeded item-waves (items ranked by
``sha256(WAVE_SALT|partition|item)``); every bank node of an item sits in one graph, hence one wave.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PASSK_SEED = 20260925
WAVE_SALT = f"passk_cont_v1/wave|seed={PASSK_SEED}"
SPLITS_VERSION = "passk-splits-v1"
SMOKE_REPLICATE = 100
SMOKE_PER_BENCHMARK = 3

PARTITIONS = ("smoke", "e1_fill", "c48_extra", "v48_bank", "t200_bank", "t200_diag")
STAGE1 = ("c48_extra", "e1_fill", "t200_diag")
STAGE2 = ("v48_bank", "t200_bank")

_BANK: tuple[tuple[int, tuple[int, ...]], ...] = (
    (16, (0,)), (8, tuple(range(2))), (4, tuple(range(4))), (2, tuple(range(8))), (1, tuple(range(16))))


@dataclass(frozen=True)
class PartitionPlan:
    name: str
    panel: str
    cells: tuple[tuple[int, tuple[int, ...]], ...]      # (multiplier of A, replicate indices), descending budget

    @property
    def per_item(self) -> int:
        return sum(len(r) for _, r in self.cells)


PLANS: dict[str, PartitionPlan] = {
    "smoke": PartitionPlan("smoke", "D32", ((1, (SMOKE_REPLICATE,)),)),
    "e1_fill": PartitionPlan("e1_fill", "L96", ((8, (1,)), (4, (1, 2, 3)), (2, tuple(range(1, 8))))),
    "c48_extra": PartitionPlan("c48_extra", "C48",
                               ((16, tuple(range(1, 8))), (8, tuple(range(2, 8))), (4, tuple(range(4, 8))))),
    "v48_bank": PartitionPlan("v48_bank", "V48", _BANK),
    "t200_bank": PartitionPlan("t200_bank", "T200", _BANK),
    "t200_diag": PartitionPlan("t200_diag", "T200", ((1, tuple(range(5))),)),
}
EXPECTED_ITEMS = {"smoke": 12, "e1_fill": 96, "c48_extra": 48, "v48_bank": 48, "t200_bank": 200, "t200_diag": 200}
EXPECTED_COUNTS = {p: EXPECTED_ITEMS[p] * PLANS[p].per_item for p in PARTITIONS}
# = {"smoke": 12, "e1_fill": 1056, "c48_extra": 816, "v48_bank": 1488, "t200_bank": 6200, "t200_diag": 1000}


class BankError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------------------------------------------------------

def artifacts_root(paths_or_dir: Any) -> Path:
    """The harness run root (``.../passk_cont_v1/artifacts``) of a shard dir or of the root itself."""
    run_dir = Path(getattr(paths_or_dir, "run_dir", paths_or_dir))
    if run_dir.parent.name == "shards":
        return run_dir.parent.parent
    return run_dir


def study_root(paths_or_dir: Any) -> Path:
    """``.../passk_cont_v1`` (holds ``sealed/``, ``gates/``, ``analysis_inputs/``): the parent of ``artifacts``."""
    return artifacts_root(paths_or_dir).parent


def splits_path(paths_or_dir: Any) -> Path:
    return artifacts_root(paths_or_dir) / "manifests" / "splits.json"


def load_splits(paths_or_dir: Any, *, path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path is not None else splits_path(paths_or_dir)
    if not source.is_file():
        raise BankError(f"no splits manifest at {source} (Gate S writes it)")
    data = json.loads(source.read_text())
    if data.get("version") != SPLITS_VERSION:
        raise BankError(f"{source}: version {data.get('version')!r} != {SPLITS_VERSION!r}")
    missing = [p for p in ("L96", "C48", "V48", "T200", "D32") if p not in data.get("panels", {})]
    if missing:
        raise BankError(f"{source}: panels {missing} missing")
    return data


def splits_sha256(paths_or_dir: Any, *, path: str | Path | None = None) -> str:
    source = Path(path) if path is not None else splits_path(paths_or_dir)
    return sha256(source.read_bytes()).hexdigest()


def benchmark_of(item_id: str) -> str:
    return item_id.split("/", 1)[0]


def smoke_items(d32: Sequence[str]) -> list[str]:
    """The first ``SMOKE_PER_BENCHMARK`` items of each benchmark's D32 list (benchmarks in contracts order)."""
    from agent_allocation.contracts import BENCHMARKS
    out: list[str] = []
    order = list(BENCHMARKS) + sorted({benchmark_of(i) for i in d32} - set(BENCHMARKS))
    for benchmark in order:
        out.extend([i for i in d32 if benchmark_of(i) == benchmark][:SMOKE_PER_BENCHMARK])
    return out


def partition_items(splits: Mapping[str, Any], partition: str) -> list[str]:
    """Every item of a partition across all shards, in panel order."""
    plan = PLANS[partition]
    items = list(splits["panels"][plan.panel])
    if partition == "smoke":
        items = smoke_items(items)
    if len(set(items)) != len(items):
        raise BankError(f"panel {plan.panel} repeats an item")
    return items


def parse_partitions(values: Iterable[str] | str) -> list[str]:
    """``'a+b'``, ``'a,b'`` or repeated values -> ordered unique partition names (validated)."""
    raw = [values] if isinstance(values, str) else list(values)
    out: list[str] = []
    for value in raw:
        for name in value.replace(",", "+").split("+"):
            name = name.strip()
            if not name:
                continue
            if name not in PLANS:
                raise BankError(f"unknown partition {name!r}; expected one of {PARTITIONS}")
            if name not in out:
                out.append(name)
    if not out:
        raise BankError("no partitions given")
    return out


# ------------------------------------------------------------------------------------------------------------------
# Nodes and graphs
# ------------------------------------------------------------------------------------------------------------------

def item_nodes(cfg: Mapping[str, Any], partition: str, task, lock, amendments: Mapping[str, int]) -> list:
    from agent_allocation.contracts import GenerationKind
    from agent_allocation.live import standalone_spec
    main_a = int(cfg["reasoning"]["main_a"])
    nodes = []
    for multiplier, replicates in PLANS[partition].cells:
        kind = GenerationKind.ROOT if multiplier == 1 else GenerationKind.DEEP
        for r in replicates:
            nodes.append(standalone_spec(cfg=cfg, partition=partition, task=task, lock=lock, kind=kind,
                                         budget=multiplier * main_a, root_slot=r, draw_id=r, amendments=amendments))
    return nodes


def build_partition(cfg: Mapping[str, Any], partition: str, items: Sequence[str], tasks: Mapping[str, Any], lock,
                    amendments: Mapping[str, int]) -> list:
    from agent_allocation.live import nodes_only_graph
    missing = [i for i in items if i not in tasks]
    if missing:
        raise BankError(f"{partition}: {len(missing)} items have no public task, e.g. {missing[:3]}")
    return [nodes_only_graph(tasks[i], partition, item_nodes(cfg, partition, tasks[i], lock, amendments),
                             f"passk_{partition}") for i in items]


def build_plan(cfg: Mapping[str, Any], splits: Mapping[str, Any], tasks: Mapping[str, Any], lock,
               amendments: Mapping[str, int], partitions: Sequence[str], *,
               allowed: set[str] | None = None) -> dict[str, list]:
    """``{partition: [ItemGraph]}`` restricted to ``allowed`` items (a shard), each partition in panel order."""
    out = {}
    for partition in partitions:
        items = [i for i in partition_items(splits, partition) if allowed is None or i in allowed]
        out[partition] = build_partition(cfg, partition, items, tasks, lock, amendments)
    return out


def wave_rank(partition: str, item_id: str) -> str:
    return sha256(f"{WAVE_SALT}|{partition}|{item_id}".encode()).hexdigest()


def plan_waves(graphs: Mapping[str, Sequence], wave_items: int) -> list[list]:
    """Stage-1 waves (``c48_extra``, ``e1_fill``, ``t200_diag`` in that order, chunked) then stage-2 seeded item-waves
    over ``v48_bank`` + ``t200_bank``. ``smoke`` (never mixed with the others) forms its own waves first."""
    if wave_items < 1:
        raise BankError("wave_items must be >= 1")
    unknown = set(graphs) - set(PARTITIONS)
    if unknown:
        raise BankError(f"unknown partitions {sorted(unknown)}")

    def chunk(seq: Sequence) -> list[list]:
        return [list(seq[k:k + wave_items]) for k in range(0, len(seq), wave_items)]

    waves: list[list] = []
    waves.extend(chunk(list(graphs.get("smoke", ()))))
    waves.extend(chunk([g for p in STAGE1 for g in graphs.get(p, ())]))
    stage2 = sorted(((wave_rank(g.partition_id, g.item_id), g.partition_id, g.item_id, g)
                     for p in STAGE2 for g in graphs.get(p, ())), key=lambda t: t[:3])
    waves.extend(chunk([t[3] for t in stage2]))
    return waves


def stage_of(partition: str) -> int:
    return 0 if partition == "smoke" else 1 if partition in STAGE1 else 2


def jobs_text_sha(graphs: Sequence) -> str:
    """sha256 of the exact ``manifests/jobs/<partition>.jsonl`` text :func:`live.write_jobs` writes for ``graphs``."""
    from agent_allocation.contracts import dump_record
    lines = [json.dumps({"node_id": n.node_id, "record": dump_record(n)}, sort_keys=True)
             for g in graphs for n in g.nodes]
    return sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def node_counts(graphs: Mapping[str, Sequence]) -> dict[str, int]:
    return {p: sum(len(g.nodes) for g in gs) for p, gs in graphs.items()}


def check_unique(graphs: Mapping[str, Sequence]) -> int:
    ids = [n.node_id for gs in graphs.values() for g in gs for n in g.nodes]
    if len(ids) != len(set(ids)):
        raise BankError(f"{len(ids) - len(set(ids))} duplicate node ids across the plan")
    return len(ids)


def legacy_node_ids(legacy_root: str | Path) -> set[str]:
    """Every node id in the legacy run root's job manifests (read-only)."""
    out: set[str] = set()
    root = Path(legacy_root)
    for jobs in sorted([*root.glob("shards/*/manifests/jobs/*.jsonl"), *root.glob("manifests/jobs/*.jsonl")]):
        with open(jobs, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.add(json.loads(line)["node_id"])
    return out


def replicate_map(graphs: Mapping[str, Sequence]) -> dict[str, dict[int, list[int]]]:
    """``{partition: {budget: sorted replicates}}`` read back from built nodes (for checks and reports)."""
    out: dict[str, dict[int, set[int]]] = {}
    for partition, gs in graphs.items():
        for g in gs:
            for n in g.nodes:
                out.setdefault(partition, {}).setdefault(n.reasoning_budget, set()).add(n.root_slot)
    return {p: {b: sorted(r) for b, r in sorted(m.items())} for p, m in out.items()}
