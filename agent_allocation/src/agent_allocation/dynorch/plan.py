"""WP-B: run planning -- (arm, item, rung, repeat, attempt) -> ``RunSpec`` (contract §6; plan §1, §6).

Inputs come read-only from the sealed forced_budget_v2 run root (``BaseInputs.load``): items from
``manifests/panels.json`` (``PanelDraw.main_item_ids`` = 96, ``dev_item_ids`` = 32), tasks from
``manifests/public/<bench>.jsonl``, golds from ``private/<bench>.jsonl``, the per-item provider shard from
``shards/assignment.json`` and each shard's pinned endpoint tag from ``shards/<shard>/locks/openrouter_endpoints.json``.

Rules enforced here (frozen table + plan §1):

* ``run_key = sha256(f"{study_id}|{arm}|{item_id}|{rung}|{repeat}|{attempt}")[:24]`` -- deterministic, no timestamp;
* ``cap_tokens = rung * a_tokens`` with ``rung`` in the config's ``dynorch.rungs``;
* order = seeded item interleave (items sorted by ``sha256(f"{seed}|{item_id}")``, so benchmarks mix), rungs inside
  each item in ``cell_rung_order`` (16, 4, 8, 25, 2, 1) so a budget or time-limit stop trims the least informative
  cells first, arms innermost so every arm loses the same cells;
* the tool surface per arm is the frozen ``allowed_tools`` list of the config arm (contract §10.3; a missing key
  raises -- the surface is never inferred) and ``spawn_tools`` likewise; a ``spawning_disabled`` arm offers no spawn
  tool; the E0 capped arms carry ``root_call_ceiling = max(A, C // 4)`` (the config spells it ``"max(A, C/4)"``; any
  other spelling raises); ``ALLOWED_TOOLS_BY_HARNESS`` documents the per-harness surfaces and is what the fake
  harness's test arms use;
* stage subsets (``plan_stage``): ``canary`` = the first dev item of each ``canary.items`` benchmark × core arms at
  ``canary.rung`` plus the k=16 ceiling probe (H1, H3); ``pilot`` = the 32 dev items × {H1, H3} × ``pilot.rungs``
  plus H0 at ``pilot.h0_rung`` on the first ``h0_items_per_benchmark`` dev items per benchmark; ``main`` = 96 main
  items × core arms × all rungs; ``ext_e0`` = H1c/H3c at {4, 16}; ``ext_e1`` = H1/H3 repeat 2 at every rung;
  ``ext_e2`` = H1n/H3n at {4, 16}; ``ext_e3`` = H2 (OpenClaw) at {4, 16}.

``write_plan`` pre-registers the planned cells as ``<run_root>/manifests/dynorch_plan_<stage>.jsonl`` (one JSON
``RunSpec`` per line, ``replace_atomic``) BEFORE a stage starts; WP-C's tables emit one ``policy_outcomes`` row per
line with ``delivered_state = "pending"`` so no cell is ever missing.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent_allocation.api import forced
from agent_allocation.contracts import BENCHMARKS, PanelDraw, PrivateEvalSpec, PublicTask, canonical_hash, load_record
from agent_allocation.dynorch.records import RunSpec
from agent_allocation.storage._atomic import replace_atomic

__all__ = [
    "ALLOWED_TOOLS_BY_HARNESS", "BaseInputs", "PROMPT_VERSION", "STAGES", "cell_rung_order", "plan_path", "plan_runs",
    "plan_stage", "read_plan", "run_key_for", "write_plan", "arm_config", "item_slug",
]

PROMPT_VERSION = "dynorch-v1"
STAGES: tuple[str, ...] = ("canary", "pilot", "main", "ext_e0", "ext_e1", "ext_e2", "ext_e3")
#: Tool surface per harness (plan §1 table: delegation only; contract §10.3 for the verified H1/H3 lists). The
#: frozen config's per-arm ``allowed_tools`` is what ``plan_runs`` stamps on a ``RunSpec``; this table documents the
#: expected surfaces and supplies the fake harness's. A request offering any other tool name is a
#: ``tool_surface_violation`` (400) -- fail closed, never silently widened.
ALLOWED_TOOLS_BY_HARNESS: Mapping[str, tuple[str, ...]] = {
    "openclaude": ("Agent",),
    "codex": ("update_plan", "spawn_agent", "send_input", "wait_agent", "resume_agent", "close_agent"),
    "openclaw": ("sessions_spawn", "subagents"),
    "null": (),
    "fake": ("spawn_subagent",),
}


def item_slug(item_id: str) -> str:
    """``aime/2025/30`` -> ``aime__2025__30`` (the per-run directory name)."""
    return item_id.replace("/", "__")


def run_key_for(study_id: str, arm: str, item_id: str, rung: int, repeat: int, attempt: int) -> str:
    return hashlib.sha256(f"{study_id}|{arm}|{item_id}|{rung}|{repeat}|{attempt}".encode("utf-8")).hexdigest()[:24]


def cell_rung_order(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    order = tuple(int(r) for r in cfg["dynorch"]["cell_rung_order"])
    rungs = tuple(int(r) for r in cfg["dynorch"]["rungs"])
    if sorted(order) != sorted(rungs):
        raise ValueError(f"cell_rung_order {order} is not a permutation of rungs {rungs}")
    return order


def arm_config(cfg: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """The arm's config block from ``dynorch.arms`` or ``dynorch.extension_arms``; unknown arm raises."""
    block = cfg["dynorch"]
    if arm in block["arms"]:
        return dict(block["arms"][arm])
    if arm in block.get("extension_arms", {}):
        return dict(block["extension_arms"][arm])
    raise KeyError(f"unknown arm {arm!r}; known: {sorted(block['arms'])} + {sorted(block.get('extension_arms', {}))}")


def _root_call_ceiling(rule: Any, *, a_tokens: int, cap: int) -> int | None:
    if rule is None:
        return None
    if rule == "max(A, C/4)":
        return max(a_tokens, cap // 4)
    raise ValueError(f"unsupported root_call_ceiling rule {rule!r} (only null or 'max(A, C/4)')")


@dataclass(frozen=True)
class BaseInputs:
    """Read-only inputs from the sealed base run root."""
    panels: Mapping[str, PanelDraw]
    tasks: Mapping[str, PublicTask]
    privates: Mapping[str, PrivateEvalSpec]
    assignment: Mapping[str, str]
    endpoint_tags: Mapping[str, str]
    model: str

    @classmethod
    def load(cls, base_run_root: str | Path, *, model: str) -> "BaseInputs":
        root = Path(base_run_root)
        panels_raw = json.loads((root / "manifests" / "panels.json").read_text("utf-8"))
        panels = {bench: load_record(PanelDraw, raw) for bench, raw in panels_raw.items()}
        tasks: dict[str, PublicTask] = {}
        privates: dict[str, PrivateEvalSpec] = {}
        for bench in BENCHMARKS:
            for line in (root / "manifests" / "public" / f"{bench}.jsonl").read_text("utf-8").splitlines():
                if line.strip():
                    task = load_record(PublicTask, json.loads(line))
                    tasks[task.item_id] = task
            for line in (root / "private" / f"{bench}.jsonl").read_text("utf-8").splitlines():
                if line.strip():
                    private = load_record(PrivateEvalSpec, json.loads(line))
                    privates[private.item_id] = private
        assignment = json.loads((root / "shards" / "assignment.json").read_text("utf-8"))["assignment"]
        if not isinstance(assignment, dict):
            raise ValueError("shards/assignment.json: 'assignment' must be an object")
        tags: dict[str, str] = {}
        for shard_id in sorted(set(assignment.values())):
            lock = json.loads((root / "shards" / shard_id / "locks" / "openrouter_endpoints.json").read_text("utf-8"))
            entry = lock["locks"].get(model)
            if entry is None:
                raise KeyError(f"shard {shard_id}: no endpoint lock for {model}")
            tags[shard_id] = str(entry["provider_tag"])
        return cls(panels=panels, tasks=tasks, privates=privates, assignment=assignment, endpoint_tags=tags, model=model)

    def main_items(self) -> tuple[str, ...]:
        return tuple(i for bench in BENCHMARKS for i in self.panels[bench].main_item_ids)

    def dev_items(self, *, per_benchmark: int | None = None) -> tuple[str, ...]:
        out: list[str] = []
        for bench in BENCHMARKS:
            ids = self.panels[bench].dev_item_ids
            out.extend(ids if per_benchmark is None else ids[:per_benchmark])
        return tuple(out)

    def task(self, item_id: str) -> PublicTask:
        return self.tasks[item_id]


def _interleave(items: Sequence[str], seed: int) -> list[str]:
    return sorted(dict.fromkeys(items), key=lambda i: hashlib.sha256(f"{seed}|{i}".encode("utf-8")).hexdigest())


def plan_runs(cfg: Mapping[str, Any], *, stage: str, arms: Sequence[str], rungs: Sequence[int],
              repeats: Sequence[int], items: Sequence[str], base: BaseInputs, proxy_version: str = "unrecorded",
              harness_versions: Mapping[str, str] | None = None) -> list[RunSpec]:
    """Every (item, rung, arm, repeat) cell as a ``RunSpec`` with ``attempt = 1`` (contract §6)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    block = cfg["dynorch"]
    a_tokens = int(block["a_tokens"])
    study_id = str(cfg["study_id"])
    known_rungs = {int(r) for r in block["rungs"]}
    bad = [r for r in rungs if int(r) not in known_rungs]
    if bad:
        raise ValueError(f"rungs {bad} are not in the config's rungs {sorted(known_rungs)}")
    rung_order = [r for r in cell_rung_order(cfg) if r in {int(x) for x in rungs}]
    sampling_hash = canonical_hash(cfg["sampling"])
    template_sha = forced.template_sha256(cfg["forced"]["template"])
    versions = dict(harness_versions or {})
    specs: list[RunSpec] = []
    for item_id in _interleave(items, int(cfg["seed"])):
        task = base.tasks.get(item_id)
        if task is None:
            raise KeyError(f"item {item_id} has no public task in the base run root")
        if item_id not in base.privates:
            raise KeyError(f"item {item_id} has no private spec in the base run root")
        shard_id = base.assignment.get(item_id)
        if shard_id is None:
            raise KeyError(f"item {item_id} has no shard in assignment.json")
        for rung in rung_order:
            cap = rung * a_tokens
            for arm in arms:
                arm_cfg = arm_config(cfg, arm)
                harness = str(arm_cfg["harness"])
                if "allowed_tools" not in arm_cfg:
                    raise KeyError(f"arm {arm}: the frozen config carries no allowed_tools (contract §10.3)")
                allowed = tuple(str(t) for t in arm_cfg["allowed_tools"])
                spawn = tuple(str(t) for t in arm_cfg["spawn_tools"])
                if not set(spawn) <= set(allowed):
                    raise ValueError(f"arm {arm}: spawn_tools {spawn} not within allowed_tools {allowed}")
                if arm_cfg.get("spawning_disabled"):
                    if spawn or set(allowed) & ALLOWED_SPAWN_NAMES:
                        raise ValueError(f"arm {arm}: spawning_disabled but a spawn tool is offered ({spawn}, {allowed})")
                ceiling = _root_call_ceiling(arm_cfg.get("root_call_ceiling"), a_tokens=a_tokens, cap=cap)
                for repeat in repeats:
                    specs.append(RunSpec(
                        run_key=run_key_for(study_id, arm, item_id, rung, int(repeat), 1), arm=arm, harness=harness,
                        item_id=item_id, benchmark=task.benchmark, stratum=task.benchmark, shard_id=shard_id,
                        endpoint_tag=base.endpoint_tags[shard_id], rung=rung, cap_tokens=cap, repeat=int(repeat),
                        attempt=1, generation=1, partition=stage, allowed_tools=allowed, spawn_tools=spawn,
                        root_call_ceiling=ceiling, sampling_hash=sampling_hash, template_sha256=template_sha,
                        prompt_version=PROMPT_VERSION, proxy_version=proxy_version,
                        harness_version=versions.get(harness, "unrecorded")))
    return specs


#: Every spawn tool name any harness offers (used to strip them for spawning-disabled arms).
ALLOWED_SPAWN_NAMES: frozenset[str] = frozenset({"Agent", "spawn_agent", "sessions_spawn", "spawn_subagent"})


def plan_stage(cfg: Mapping[str, Any], stage: str, base: BaseInputs, *, arms: Sequence[str] | None = None,
               rungs: Sequence[int] | None = None, proxy_version: str = "unrecorded",
               harness_versions: Mapping[str, str] | None = None) -> list[RunSpec]:
    """The frozen cell set of one stage (module docstring); ``arms``/``rungs`` narrow it (CLI filters)."""
    block = cfg["dynorch"]
    core_arms = [str(a) for a in block["core_arms"]]
    all_rungs = [int(r) for r in block["rungs"]]
    kw = dict(cfg=cfg, stage=stage, base=base, proxy_version=proxy_version, harness_versions=harness_versions)

    def _runs(arm_list: Sequence[str], rung_list: Sequence[int], items: Sequence[str], repeats=(1,)) -> list[RunSpec]:
        chosen_arms = [a for a in arm_list if arms is None or a in arms]
        chosen_rungs = [r for r in rung_list if rungs is None or r in {int(x) for x in rungs}]
        if not chosen_arms or not chosen_rungs:
            return []
        return plan_runs(arms=chosen_arms, rungs=chosen_rungs, repeats=repeats, items=items, **kw)

    harness_arms = [a for a in core_arms if arm_config(cfg, a)["harness"] != "null"]
    null_arms = [a for a in core_arms if arm_config(cfg, a)["harness"] == "null"]
    if stage == "canary":
        canary = block["canary"]
        items = [base.panels[b].dev_item_ids[0] for b in canary["items"]]
        specs = _runs(core_arms, [int(canary["rung"])], items)
        probe_items = base.dev_items(per_benchmark=1)[: int(canary["ceiling_probe_items"])]
        specs += _runs(harness_arms, [int(canary["ceiling_probe_rung"])], probe_items)
        return _dedupe(specs)
    if stage == "pilot":
        pilot = block["pilot"]
        dev = base.dev_items(per_benchmark=int(pilot["dev_items_per_benchmark"]))
        specs = _runs(harness_arms, [int(r) for r in pilot["rungs"]], dev)
        specs += _runs(null_arms, [int(pilot["h0_rung"])], base.dev_items(per_benchmark=int(pilot["h0_items_per_benchmark"])))
        return _dedupe(specs)
    main_items = base.main_items()
    if stage == "main":
        return _runs(core_arms, all_rungs, main_items)
    if stage == "ext_e0":
        return _runs(["H1c", "H3c"], [4, 16], main_items)
    if stage == "ext_e1":
        return _runs(harness_arms, all_rungs, main_items, repeats=(2,))
    if stage == "ext_e2":
        return _runs(["H1n", "H3n"], [4, 16], main_items)
    if stage == "ext_e3":
        return _runs(["H2"], [4, 16], main_items)
    raise ValueError(f"unknown stage {stage!r}")


def _dedupe(specs: Iterable[RunSpec]) -> list[RunSpec]:
    seen: set[str] = set()
    out: list[RunSpec] = []
    for spec in specs:
        if spec.run_key not in seen:
            seen.add(spec.run_key)
            out.append(spec)
    return out


# ---------------------------------------------------------------------------------------------------------
# Pre-registration file
# ---------------------------------------------------------------------------------------------------------

def plan_path(run_root: str | Path, stage: str) -> Path:
    return Path(run_root) / "manifests" / f"dynorch_plan_{stage}.jsonl"


def spec_to_json(spec: RunSpec) -> dict[str, Any]:
    return dataclasses.asdict(spec)


def spec_from_json(data: Mapping[str, Any]) -> RunSpec:
    fields = {f.name for f in dataclasses.fields(RunSpec)}
    unknown = set(data) - fields
    missing = fields - set(data)
    if unknown or missing:
        raise ValueError(f"RunSpec json: unknown {sorted(unknown)} missing {sorted(missing)}")
    body = dict(data)
    body["allowed_tools"] = tuple(body["allowed_tools"])
    body["spawn_tools"] = tuple(body["spawn_tools"])
    return RunSpec(**body)


def write_plan(specs: Sequence[RunSpec], run_root: str | Path, stage: str) -> Path:
    """Pre-register the stage's cells (one ``RunSpec`` per line) before it starts; idempotent."""
    path = plan_path(run_root, stage)
    lines = [json.dumps(spec_to_json(s), sort_keys=True, ensure_ascii=False) for s in specs]
    replace_atomic(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    return path


def read_plan(path: str | Path) -> list[RunSpec]:
    return [spec_from_json(json.loads(line)) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
