"""Shared source handling and the seeded, outcome-blind panel draw for the three text benchmarks.

Config keys consumed (R9): ``benchmarks[id].source`` and ``benchmarks[id].split`` / ``.subtasks`` are checked
against the pinned sources by :func:`benchmark_config`; ``seed`` is passed to :func:`select_panels` by the
lead's ``prepare-data``. Panel sizes arrive as arguments (``n_dev``/``n_main``/``n_reserve``).

Panel procedure ``text-panels-v1`` (frozen; changing any rule below requires a new version string):

* Population: the benchmark's ``InventoryItem``s with ``eligible=True``. Ineligible items keep their
  pre-generation ``exclusion_reason`` and are never drawn.
* Rank: within a stratum, items are ordered by ``sha256(salt | "rank" | item_id)`` with
  ``salt = "<STUDY_ID>|<benchmark>|seed=<seed>|text-panels-v1"``. No model outcome enters any key.
* MATH-500 / MMLU-Pro: a stratum is the item's full ``strata`` tuple (MATH ``level`` x ``subject``, MMLU
  ``category``). ``main`` is allocated first by exact largest-remainder (Hamilton) apportionment of ``n_main``
  over stratum sizes (remainder ties broken by a seeded stratum hash, no minimum per stratum); each stratum
  gives its first ranked items. ``dev`` is then apportioned the same way over what remains, then ``reserve``.
  Order inside each panel is a seeded systematic interleave: the ``j``-th of ``a`` chosen items of a stratum
  sits at ``(j + u) / a`` with a seeded offset ``u`` in [0, 1), so every wave prefix stays close to the
  panel's stratum proportions.
* BBH-logic4: per subtask, ranked items give ``main`` (``n_main/4``), then ``dev`` (``n_dev/4``), then
  ``reserve`` (``n_reserve/4``). Panels are laid out in blocks of four, one per subtask in ``BBH_SUBTASKS``
  order, so ``main_item_ids[4w:4w+4]`` is exactly one item per subtask.

Dev, main and reserve are disjoint by construction; nested panels are prefixes of ``main_item_ids``.
"""
from __future__ import annotations

import collections
import hashlib
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent_allocation.contracts import BBH_SUBTASKS, STUDY_ID, InventoryItem, PanelDraw

PANEL_PROCEDURE_VERSION = "text-panels-v1"
DEFAULT_HF_HOME = Path("/orcd/data/tpoggio/001/mabdel03/.cache/huggingface")
TEXT_BENCHMARKS: tuple[str, ...] = ("math500", "mmlu_pro", "bbh_logic4")
_STRATUM_RULE = {
    "math500": "level x subject",
    "mmlu_pro": "category",
    "bbh_logic4": "subtask (equal)",
}


class SourceIntegrityError(RuntimeError):
    """A pinned source is missing, changed, or does not have the frozen shape. Never silently replaced."""


# ---------------------------------------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or DEFAULT_HF_HOME)


def hf_snapshot_file(repo: str, revision: str, relpath: str) -> Path:
    """Path of a file inside the offline Hugging Face hub cache snapshot for ``repo@revision``."""
    return hf_home() / "hub" / f"datasets--{repo.replace('/', '--')}" / "snapshots" / revision / relpath


def read_verified(path: Path, expected_sha256: str) -> bytes:
    """The bytes of ``path`` if and only if their sha256 is the pinned value."""
    if not path.is_file():
        raise SourceIntegrityError(f"pinned source file is missing: {path}")
    data = path.read_bytes()
    observed = sha256_bytes(data)
    if observed != expected_sha256:
        raise SourceIntegrityError(f"{path}: sha256 {observed} != pinned {expected_sha256}")
    return data


def benchmark_config(cfg: Mapping[str, Any], benchmark: str, *, source: str) -> Mapping[str, Any]:
    """The config entry for ``benchmark``; its ``source`` must be the pinned upstream."""
    entries = [entry for entry in cfg.get("benchmarks", ()) if entry.get("id") == benchmark]
    if len(entries) != 1:
        raise ValueError(f"config must declare benchmark {benchmark!r} exactly once")
    entry = entries[0]
    if entry.get("source") != source:
        raise ValueError(f"{benchmark}: config source {entry.get('source')!r} != pinned {source!r}")
    return entry


# ---------------------------------------------------------------------------------------------------------
# Seeded ranks and apportionment
# ---------------------------------------------------------------------------------------------------------

def rank_key(salt: str, *parts: str) -> str:
    return hashlib.sha256("\x1f".join((salt, *parts)).encode("utf-8")).hexdigest()


def _unit_interval(salt: str, *parts: str) -> Fraction:
    return Fraction(int(rank_key(salt, *parts)[:15], 16), 16 ** 15)


def panel_salt(benchmark: str, seed: int) -> str:
    return f"{STUDY_ID}|{benchmark}|seed={int(seed)}|{PANEL_PROCEDURE_VERSION}"


def largest_remainder(sizes: Mapping[str, int], total: int, tie_key: Callable[[str], str]) -> dict[str, int]:
    """Exact Hamilton apportionment of ``total`` over ``sizes`` (no minimum per stratum, capped by size)."""
    population = sum(sizes.values())
    if total < 0 or total > population:
        raise ValueError(f"cannot draw {total} items from a population of {population}")
    if total == 0:
        return {name: 0 for name in sizes}
    quotas = {name: Fraction(total * size, population) for name, size in sizes.items()}
    alloc = {name: int(quota) for name, quota in quotas.items()}          # floor; quota <= size
    order = sorted(sizes, key=lambda name: (-(quotas[name] - alloc[name]), tie_key(name)))
    missing = total - sum(alloc.values())
    for name in order:
        if missing == 0:
            break
        if alloc[name] < sizes[name]:
            alloc[name] += 1
            missing -= 1
    if missing != 0:                                                      # impossible for exact quotas
        raise AssertionError("largest-remainder apportionment did not reach its total")
    return alloc


def _stratum_name(item: InventoryItem) -> str:
    return "|".join(f"{key}={value}" for key, value in item.strata)


def _interleave(groups: Mapping[str, Sequence[str]], salt: str, label: str) -> list[str]:
    keyed: list[tuple[Fraction, str, str]] = []
    for name in sorted(groups):
        chosen = groups[name]
        if not chosen:
            continue
        offset = _unit_interval(salt, label, "offset", name)
        for j, item_id in enumerate(chosen):
            keyed.append(((j + offset) / len(chosen), rank_key(salt, label, "tie", item_id), item_id))
    keyed.sort()
    return [item_id for _, _, item_id in keyed]


# ---------------------------------------------------------------------------------------------------------
# Panel draw
# ---------------------------------------------------------------------------------------------------------

def _check_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int, got {value!r}")
    return value


def _benchmark_items(inventory: Sequence[InventoryItem], benchmark: str) -> list[InventoryItem]:
    items = [item for item in inventory if item.public.benchmark == benchmark]
    if not items:
        raise ValueError(f"inventory holds no {benchmark} items")
    ids = [item.public.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{benchmark}: duplicate item ids in inventory")
    for item in items:
        if item.private.item_id != item.public.item_id:
            raise ValueError(f"{item.public.item_id}: public/private item ids differ")
    return items


def _stratified_panels(eligible: Sequence[InventoryItem], salt: str, sizes: Mapping[str, int]
                       ) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in eligible:
        groups.setdefault(_stratum_name(item), []).append(item.public.item_id)
    remaining = {name: sorted(ids, key=lambda item_id: rank_key(salt, "rank", item_id))
                 for name, ids in groups.items()}
    panels: dict[str, list[str]] = {}
    for panel in ("main", "dev", "reserve"):                               # main first: stable under n_dev
        alloc = largest_remainder({name: len(ids) for name, ids in remaining.items()}, sizes[panel],
                                  lambda name, panel=panel: rank_key(salt, panel, "stratum", name))
        chosen = {name: remaining[name][:alloc[name]] for name in remaining}
        remaining = {name: remaining[name][alloc[name]:] for name in remaining}
        panels[panel] = _interleave(chosen, salt, panel)
    return panels


def _bbh_panels(eligible: Sequence[InventoryItem], salt: str, sizes: Mapping[str, int]) -> dict[str, list[str]]:
    per_subtask: dict[str, list[str]] = {subtask: [] for subtask in BBH_SUBTASKS}
    for item in eligible:
        if item.public.subtask not in per_subtask:
            raise ValueError(f"{item.public.item_id}: subtask {item.public.subtask!r} is not in BBH_SUBTASKS")
        per_subtask[item.public.subtask].append(item.public.item_id)
    width = len(BBH_SUBTASKS)
    for panel, size in sizes.items():
        if size % width:
            raise ValueError(f"bbh_logic4 {panel} size {size} is not a multiple of {width}")
    needed = sum(size // width for size in sizes.values())
    ranked = {}
    for subtask, ids in per_subtask.items():
        if len(ids) < needed:
            raise ValueError(f"bbh_logic4/{subtask}: {len(ids)} eligible items < {needed} required")
        ranked[subtask] = sorted(ids, key=lambda item_id: rank_key(salt, "rank", item_id))
    panels: dict[str, list[str]] = {}
    start = 0
    for panel in ("main", "dev", "reserve"):
        per = sizes[panel] // width
        panels[panel] = [ranked[subtask][start + w] for w in range(per) for subtask in BBH_SUBTASKS]
        start += per
    return panels


def select_panels(inventory: Sequence[InventoryItem], *, benchmark: str, seed: int, n_dev: int, n_main: int,
                  n_reserve: int) -> PanelDraw:
    """Seeded, outcome-blind dev/main/reserve draw for one text benchmark (procedure in the module docstring)."""
    if benchmark not in TEXT_BENCHMARKS:
        raise ValueError(f"select_panels handles {TEXT_BENCHMARKS}, not {benchmark!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    sizes = {"main": _check_count("n_main", n_main), "dev": _check_count("n_dev", n_dev),
             "reserve": _check_count("n_reserve", n_reserve)}
    items = _benchmark_items(inventory, benchmark)
    eligible = [item for item in items if item.eligible]
    if sum(sizes.values()) > len(eligible):
        raise ValueError(f"{benchmark}: {sum(sizes.values())} panel items requested, {len(eligible)} eligible")
    salt = panel_salt(benchmark, seed)
    if benchmark == "bbh_logic4":
        panels = _bbh_panels(eligible, salt, sizes)
    else:
        panels = _stratified_panels(eligible, salt, sizes)
    draw = PanelDraw(
        benchmark=benchmark,
        dev_item_ids=tuple(panels["dev"]),
        main_item_ids=tuple(panels["main"]),
        reserve_item_ids=tuple(panels["reserve"]),
        selection_seed=int(seed),
        procedure=(f"{PANEL_PROCEDURE_VERSION}: strata={_STRATUM_RULE[benchmark]}; "
                   + ("per-subtask sha256 rank; main, dev, reserve in blocks of one per subtask in "
                      "BBH_SUBTASKS order" if benchmark == "bbh_logic4" else
                      "sha256 rank within stratum; largest-remainder apportionment main, then dev, then "
                      "reserve; seeded systematic interleave order")
                   + f"; eligible={len(eligible)} of {len(items)}"),
    )
    validate_panel_draw(draw, items)
    return draw


def validate_panel_draw(draw: PanelDraw, items: Iterable[InventoryItem]) -> None:
    """Structural invariants every draw must satisfy; raises ``ValueError`` on the first violation."""
    by_id = {item.public.item_id: item for item in items}
    panels = {"dev": draw.dev_item_ids, "main": draw.main_item_ids, "reserve": draw.reserve_item_ids}
    seen: set[str] = set()
    for name, ids in panels.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"{draw.benchmark} {name}: duplicate ids")
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"{draw.benchmark} {name}: overlaps another panel: {sorted(overlap)[:3]}")
        seen.update(ids)
        for item_id in ids:
            item = by_id.get(item_id)
            if item is None or item.public.benchmark != draw.benchmark:
                raise ValueError(f"{draw.benchmark} {name}: unknown item {item_id}")
            if not item.eligible:
                raise ValueError(f"{draw.benchmark} {name}: ineligible item {item_id}")
    if draw.benchmark == "bbh_logic4":
        width = len(BBH_SUBTASKS)
        for name, ids in panels.items():
            for start in range(0, len(ids), width):
                block = tuple(by_id[item_id].public.subtask for item_id in ids[start:start + width])
                if block != BBH_SUBTASKS:
                    raise ValueError(f"bbh_logic4 {name} block {start // width} is {block}, not BBH_SUBTASKS")


def exclusion_records(inventory: Iterable[InventoryItem]) -> list[dict[str, str]]:
    """The engineering exclusion audit: every ineligible item with its pre-generation reason, sorted by id."""
    rows = []
    for item in inventory:
        if item.eligible:
            continue
        if not item.exclusion_reason:
            raise ValueError(f"{item.public.item_id}: ineligible without an exclusion reason")
        rows.append({"benchmark": item.public.benchmark, "item_id": item.public.item_id,
                     "source_revision": item.public.source_revision, "reason": item.exclusion_reason})
    return sorted(rows, key=lambda row: (row["benchmark"], row["item_id"]))


def panel_audit(inventory: Sequence[InventoryItem], draw: PanelDraw) -> dict[str, Any]:
    """Population and per-panel stratum counts for the dataset lock and the completeness report."""
    items = _benchmark_items(inventory, draw.benchmark)
    validate_panel_draw(draw, items)
    by_id = {item.public.item_id: item for item in items}
    eligible = [item for item in items if item.eligible]

    def strata(ids: Iterable[str]) -> dict[str, int]:
        return dict(sorted(collections.Counter(_stratum_name(by_id[item_id]) for item_id in ids).items()))

    return {
        "benchmark": draw.benchmark, "selection_seed": draw.selection_seed, "procedure": draw.procedure,
        "population": {"items": len(items), "eligible": len(eligible), "excluded": len(items) - len(eligible),
                       "strata": strata(item.public.item_id for item in eligible)},
        "panels": {name: {"size": len(ids), "strata": strata(ids)}
                   for name, ids in (("dev", draw.dev_item_ids), ("main", draw.main_item_ids),
                                     ("reserve", draw.reserve_item_ids))},
    }


# ---------------------------------------------------------------------------------------------------------
# Typed synthetic fixture (for downstream agents' tests; never real benchmark items)
# ---------------------------------------------------------------------------------------------------------

FIXTURE_REVISION = "synthetic-fixture-v1"
_FIXTURE_MATH_ANSWERS = ("\\frac{1}{2}", "3", "x=5", "\\sqrt{2}", "(1,2)", "\\text{east}")


def fixture_inventory(benchmark: str) -> list[InventoryItem]:
    """A small synthetic inventory built through the real adapter ``build_item`` functions.

    ``math500``: 24 items over 3 levels x 2 subjects; ``mmlu_pro``: 24 items over 3 categories with 4-10 options;
    ``bbh_logic4``: 6 items per subtask. ``source_revision`` is ``synthetic-fixture-v1``; prompts never contain the
    gold answer, so leakage tests stay meaningful. Panels: e.g. ``n_dev=4, n_main=8, n_reserve=4``.
    """
    if benchmark == "math500":
        from agent_allocation.benchmarks import math500

        rows = [{"problem": f"Synthetic MATH fixture problem {chr(ord('a') + k)}.", "solution": "Synthetic.",
                 "answer": _FIXTURE_MATH_ANSWERS[k % len(_FIXTURE_MATH_ANSWERS)],
                 "subject": ("Algebra", "Geometry")[k % 2], "level": 1 + k % 3,
                 "unique_id": f"test/synthetic/{k}.json"} for k in range(24)]
        return [math500.build_item(row, source_revision=FIXTURE_REVISION) for row in rows]
    if benchmark == "mmlu_pro":
        from agent_allocation.benchmarks import mmlu_pro

        rows = []
        for k in range(24):
            n_options = 4 + k % 7
            rows.append({"question_id": 900000 + k, "question": f"Synthetic MMLU-Pro fixture question {k}?",
                         "options": [f"fixture option {k}.{j}" for j in range(n_options)],
                         "answer": mmlu_pro.LETTERS[k % n_options], "answer_index": k % n_options,
                         "cot_content": "", "category": ("math", "law", "health")[k % 3], "src": "synthetic"})
        items = [mmlu_pro.build_item(row, source_revision=FIXTURE_REVISION) for row in rows]
        return mmlu_pro.apply_source_item_rule(items, [row["question"] for row in rows])
    if benchmark == "bbh_logic4":
        from agent_allocation.benchmarks import bbh_logic4

        items = []
        for subtask in BBH_SUBTASKS:
            for index in range(6):
                if subtask == bbh_logic4.BOOLEAN_SUBTASK:
                    example = {"input": f"{'not ' * (index % 2)}( True and False ) or True is",
                               "target": ("True", "False")[index % 2]}
                else:
                    letters = "ABCDEFG" if subtask != "temporal_sequences" else "ABCD"
                    options = "\n".join(f"({letter}) fixture choice {letter}" for letter in letters)
                    example = {"input": f"Synthetic {subtask} fixture {index}.\nOptions:\n{options}",
                               "target": f"({letters[index % len(letters)]})"}
                items.append(bbh_logic4.build_item(subtask, index, example, source_revision=FIXTURE_REVISION))
        return items
    raise ValueError(f"no text fixture for {benchmark!r}")
