"""The single reader (contract §5.3): legacy + new-run candidates -> ``analysis_inputs/draws.parquet``.

Only this module opens ``shards/*/{parsed,raw,grades}``, the public manifests and the legacy run root. Every fit
stage reads ``analysis_inputs/`` and ``sealed/`` only.

Sources:

* **legacy** (read-only ``forced_budget_v2/artifacts``): L96 = partition ``main`` kind ``root`` at A (slots 0-24)
  and kind ``deep`` at 2/4/8/16/25A (replicate r0); D32 = partition ``pilot`` kind ``root`` at A.
* **passk** (``passk_cont_v1/artifacts/shards/<shard>``): node -> partition from ``manifests/jobs/<partition>.jsonl``.

Fence: before a verified seal the allow-list is legacy, ``smoke``, ``e1_fill``, ``c48_extra`` and ``t200_diag``
*without* ``correct``/``grade_state``. ``v48_bank``/``t200_bank`` are never opened before the seal; afterwards
``release={"v48"}`` / ``{"t200"}`` adds them (and ``t200`` releases ``t200_diag`` correctness). A release without
a verifying seal raises.

Correctness is tri-state (``metrics.outcomes.candidate_state``): invalid output -> False, grader unknown/ungraded ->
None. ``grade_state`` is one of correct / wrong / invalid / unknown / missing.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from agent_allocation.contracts import Candidate, GradeResult, OutputKind, PublicTask, load_record
from agent_allocation.metrics.outcomes import CandidateState, candidate_state
from agent_allocation.selection.keys import option_letters, vote_key

__all__ = [
    "A", "BENCHMARKS", "DRAW_COLUMNS", "FENCED_BANKS", "LEGACY_ROOT", "PRE_SEAL_PARTITIONS", "RELEASES",
    "RUN_ROOT", "extract", "legacy_draws", "load_draws", "load_public_tasks", "passk_draws", "read_jobs",
]

A = 1024
BENCHMARKS = ("aime", "hmmt", "omni_math_hard", "gpqa_diamond")
LEGACY_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/forced_budget_v2/artifacts")
RUN_ROOT = Path("/orcd/data/tpoggio/001/mabdel03/agents_scaling_results/passk_cont_v1")
SHARDS = ("mancer", "parasail")
PRE_SEAL_PARTITIONS = ("smoke", "e1_fill", "c48_extra", "t200_diag")
FENCED_BANKS = ("v48_bank", "t200_bank")
LABEL_FENCED = ("v48_bank", "t200_bank", "t200_diag")
RELEASES = {"v48": ("v48_bank",), "t200": ("t200_bank", "t200_diag")}
LABEL_COLUMNS = ("correct", "grade_state")
_STATE = {CandidateState.CORRECT: "correct", CandidateState.WRONG: "wrong", CandidateState.INVALID: "invalid",
          CandidateState.UNKNOWN: "unknown", CandidateState.MISSING: "missing"}

DRAW_COLUMNS = [
    "candidate_id", "source", "partition", "panel_roles", "draw_role", "item_id", "benchmark", "shard", "kind",
    "allowance_a", "replicate", "parse_status", "final_present", "vote_text", "vote_exact_key", "option",
    "answer_len_chars", "answer_finish_reason", "think_latency_s", "latency_s", "think_in", "think_out",
    "answer_in", "answer_out", "cost_usd", "generation_id_think", "generation_id_answer", "observed_model",
    "observed_provider", "absent", "correct", "grade_state",
    # provenance extras (E0 / reuse manifest; not labels)
    "raw_response_path", "raw_response_sha256", "think_body_sha", "answer_body_sha", "provider_only",
    "decoding_config_hash", "prompt_version", "completion_headroom", "endpoint_lock_hash", "grader_version",
]


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_public_tasks(run_dir: Path) -> dict[str, PublicTask]:
    tasks: dict[str, PublicTask] = {}
    for benchmark in BENCHMARKS:
        manifest = Path(run_dir) / "manifests" / "public" / f"{benchmark}.jsonl"
        if not manifest.exists():
            continue
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    task = load_record(PublicTask, json.loads(line))
                    tasks[task.item_id] = task
    return tasks


def read_jobs(shard_dir: Path, partition: str) -> list[tuple[str, dict]]:
    path = Path(shard_dir) / "manifests" / "jobs" / f"{partition}.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out.append((row["node_id"], row["record"]))
    return out


def _body_sha(body: dict | None, *, drop: tuple[str, ...] = ()) -> str | None:
    if not isinstance(body, dict):
        return None
    kept = {k: v for k, v in body.items() if k not in drop}
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _raw_facts(shard_dir: Path, raw_path: str | None) -> dict:
    """THINK generation id and request-body hashes (THINK without ``max_tokens``) from the raw gz record."""
    if not raw_path:
        return {}
    path = Path(shard_dir) / raw_path
    if not path.exists():
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
        body = json.loads(record.get("response_body_text") or "{}")
    except (OSError, ValueError):
        return {}
    think = body.get("think") if isinstance(body, dict) else None
    request = record.get("request_body") or {}
    think_req = request.get("think") if isinstance(request, dict) else None
    only = ((think_req or {}).get("provider") or {}).get("only")
    return {"generation_id_think": think.get("id") if isinstance(think, dict) else None,
            "think_body_sha": _body_sha(think_req, drop=("max_tokens",)),
            "answer_body_sha": _body_sha(request.get("answer") if isinstance(request, dict) else None),
            "provider_only": None if only is None else ",".join(only)}


def _usage(result: dict) -> dict:
    usage = result.get("usage") or {}
    raw = usage.get("raw_usage") or {}
    think, answer = raw.get("think") or {}, raw.get("answer") or {}
    cost = usage.get("actual_cost_usd")
    return {"think_in": think.get("prompt_tokens"), "think_out": think.get("completion_tokens"),
            "answer_in": answer.get("prompt_tokens"), "answer_out": answer.get("completion_tokens"),
            "cost_usd": None if cost is None else float(cost)}


def draw_row(shard_dir: Path, node_id: str, spec: dict, task: PublicTask | None, *, source: str, partition: str,
             shard: str, with_labels: bool, read_raw: bool = True) -> dict:
    """One ``draws.parquet`` row for a node; labels only when ``with_labels`` (the fence decides)."""
    shard_dir = Path(shard_dir)
    parsed_path = shard_dir / "parsed" / node_id[:2] / f"{node_id}.json"
    budget = int(spec["reasoning_budget"])
    row: dict[str, Any] = {c: None for c in DRAW_COLUMNS}
    row.update(candidate_id=node_id, source=source, partition=partition, item_id=spec["item_id"],
               benchmark=spec["benchmark"], shard=shard, kind=spec["kind"], allowance_a=budget // A,
               replicate=int(spec.get("draw_id", spec.get("root_slot", 0)) if source == "passk"
                             else (spec.get("root_slot", 0) if spec["kind"] == "root" else 0)),
               absent=True, final_present=False, decoding_config_hash=spec.get("decoding_config_hash"),
               prompt_version=spec.get("prompt_version"), completion_headroom=spec.get("completion_headroom"),
               endpoint_lock_hash=spec.get("endpoint_lock_hash"))
    record = _read_json(parsed_path) if parsed_path.exists() else None
    cand_rec = None if record is None else record.get("candidate")
    result = None if record is None else record.get("result")
    candidate = None if cand_rec is None else load_record(Candidate, cand_rec)
    if result:
        row.update(_usage(result))
        row["latency_s"] = result.get("latency_seconds")
        row["observed_model"] = result.get("observed_model")
        row["observed_provider"] = result.get("observed_provider")
        row["generation_id_answer"] = result.get("generation_id")
        row["raw_response_path"] = result.get("raw_response_path")
        row["raw_response_sha256"] = result.get("raw_response_sha256")
        if read_raw:
            row.update(_raw_facts(shard_dir, result.get("raw_response_path")))
    if candidate is not None:
        row["absent"] = False
        row["parse_status"] = candidate.parse_status.value
        row["final_present"] = candidate.final_answer is not None
        row["answer_finish_reason"] = candidate.finish_reason
        row["answer_len_chars"] = None if candidate.visible_content is None else len(candidate.visible_content)
        if task is not None:
            letters = option_letters(task) if task.output_kind is OutputKind.OPTION else None
            key = vote_key(candidate, task, letters=letters)
            if key is not None:
                row["vote_text"], row["vote_exact_key"] = key.text, key.exact
                if task.output_kind is OutputKind.OPTION:
                    row["option"] = key.exact
    if with_labels:
        grade_path = shard_dir / "grades" / node_id[:2] / f"{node_id}.json"
        grade = load_record(GradeResult, _read_json(grade_path)) if grade_path.exists() else None
        if candidate is None:
            grade = None
        state = candidate_state(candidate, grade)
        row["grader_version"] = None if grade is None else grade.grader_version
        row["grade_state"] = _STATE[state]
        row["correct"] = {CandidateState.CORRECT: True, CandidateState.WRONG: False,
                          CandidateState.INVALID: False}.get(state)
    return row


def _splits(run_root: Path) -> dict:
    path = Path(run_root) / "artifacts" / "manifests" / "splits.json"
    return _read_json(path) if path.exists() else {}


def _roles(item_id: str, panels: dict[str, set[str]], wanted: Iterable[str]) -> list[str]:
    return [p for p in wanted if item_id in panels.get(p, set())]


def _legacy_draw_role(kind: str, slot: int) -> str:
    if kind != "root":
        return "deep_r0"
    return "pool" if slot < 16 else ("bank" if slot < 20 else "diag")


def legacy_draws(legacy_root: Path = LEGACY_ROOT, *, with_labels: bool = True, panels: dict | None = None,
                 read_raw: bool = True) -> pd.DataFrame:
    """L96 root (A, slots 0-24) + deep (2..25A) and D32 roots at A from the read-only legacy run root."""
    legacy_root = Path(legacy_root)
    tasks = load_public_tasks(legacy_root)
    panel_sets = {k: set(v) for k, v in (panels or {}).items()}
    rows = []
    for shard in SHARDS:
        shard_dir = legacy_root / "shards" / shard
        for partition, panel in (("main", "L96"), ("pilot", "D32")):
            for node_id, spec in read_jobs(shard_dir, partition):
                kind, budget = spec["kind"], int(spec["reasoning_budget"])
                keep = (kind == "root" and budget == A) or (partition == "main" and kind == "deep")
                if not keep:
                    continue
                row = draw_row(shard_dir, node_id, spec, tasks.get(spec["item_id"]), source="legacy",
                               partition=f"legacy_{partition}", shard=shard, with_labels=with_labels,
                               read_raw=read_raw)
                row["panel_roles"] = [panel] + _roles(spec["item_id"], panel_sets, ("C48", "V48"))
                row["draw_role"] = _legacy_draw_role(kind, int(spec.get("root_slot", 0)))
                rows.append(row)
    return _frame(rows)


def passk_draws(run_root: Path, partitions: Iterable[str], *, labelled: Iterable[str],
                panels: dict | None = None, read_raw: bool = True) -> pd.DataFrame:
    run_root = Path(run_root)
    art = run_root / "artifacts"
    tasks = load_public_tasks(art)
    for shard in SHARDS:
        tasks.update({k: v for k, v in load_public_tasks(art / "shards" / shard).items() if k not in tasks})
    panel_sets = {k: set(v) for k, v in (panels or {}).items()}
    labelled = set(labelled)
    rows = []
    for shard in SHARDS:
        shard_dir = art / "shards" / shard
        for partition in partitions:
            for node_id, spec in read_jobs(shard_dir, partition):
                row = draw_row(shard_dir, node_id, spec, tasks.get(spec["item_id"]), source="passk",
                               partition=partition, shard=shard, with_labels=partition in labelled,
                               read_raw=read_raw)
                row["panel_roles"] = _roles(spec["item_id"], panel_sets, ("L96", "C48", "V48", "T200", "D32",
                                                                           "X80"))
                row["draw_role"] = partition
                rows.append(row)
    return _frame(rows)


def _frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=DRAW_COLUMNS)
    for col in ("correct",):
        frame[col] = frame[col].astype("object")
    return frame


def _seal_functions():
    """``blocked_node_ids``/``verify_seal`` from WP-G's ``grade_scope`` when present, else the local shim."""
    try:
        from agent_allocation.passk.grade_scope import verify_seal  # type: ignore[attr-defined]
    except ImportError:
        from agent_allocation.passk.labels import verify_seal
    return verify_seal


def _code_commit() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        return subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class ExtractPlan:
    partitions: tuple[str, ...]
    labelled: tuple[str, ...]
    releases: tuple[str, ...]


def plan_partitions(run_root: Path, release: Iterable[str] = ()) -> ExtractPlan:
    release = tuple(sorted(set(release)))
    unknown = [r for r in release if r not in RELEASES]
    if unknown:
        raise ValueError(f"unknown release {unknown}; expected one of {sorted(RELEASES)}")
    partitions = list(PRE_SEAL_PARTITIONS)
    labelled = [p for p in PRE_SEAL_PARTITIONS if p not in LABEL_FENCED]
    if release:
        seal = _seal_functions()(run_root)
        if seal is None:
            raise PermissionError(f"{run_root}: release {release} requested without a verifying SEAL_predictions")
        for name in release:
            for partition in RELEASES[name]:
                if partition not in partitions:
                    partitions.append(partition)
                labelled.append(partition)
    return ExtractPlan(tuple(partitions), tuple(dict.fromkeys(labelled)), release)


def extract(run_root: Path = RUN_ROOT, *, legacy_root: Path = LEGACY_ROOT, release: Iterable[str] = (),
            read_raw: bool = True, include_legacy: bool = True) -> dict:
    """Write ``analysis_inputs/draws.parquet`` + ``manifest.json``; returns the manifest."""
    run_root = Path(run_root)
    plan = plan_partitions(run_root, release)
    splits = _splits(run_root)
    panels = splits.get("panels", {})
    frames = []
    if include_legacy and Path(legacy_root).exists():
        frames.append(legacy_draws(legacy_root, with_labels=True, panels=panels, read_raw=read_raw))
    frames.append(passk_draws(run_root, plan.partitions, labelled=plan.labelled, panels=panels, read_raw=read_raw))
    draws = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) \
        else _frame([])
    unlabelled = ~draws["partition"].isin(plan.labelled) & (draws["source"] == "passk")
    draws.loc[unlabelled, list(LABEL_COLUMNS)] = None
    if draws["candidate_id"].duplicated().any():
        raise ValueError("duplicate candidate ids across sources")
    out_dir = run_root / "analysis_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "draws.parquet"
    tmp = out_dir / f".draws.parquet.tmp{os.getpid()}"
    draws.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    manifest = {
        "version": "passk-extract-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _code_commit(),
        "legacy_root": str(legacy_root) if include_legacy else None,
        "partitions_read": ["legacy_main", "legacy_pilot"] * bool(include_legacy) + list(plan.partitions),
        "labelled_partitions": list(plan.labelled),
        "releases": list(plan.releases),
        "allow_list": {"pre_seal": list(PRE_SEAL_PARTITIONS), "label_fenced": list(LABEL_FENCED),
                       "fenced_banks": list(FENCED_BANKS)},
        "n_rows": int(len(draws)),
        "n_rows_by_partition": {str(k): int(v) for k, v in draws["partition"].value_counts().sort_index().items()},
        "sha256": {"draws.parquet": _sha256(target)},
    }
    tmp_manifest = out_dir / f".manifest.json.tmp{os.getpid()}"
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, out_dir / "manifest.json")
    return manifest


def load_draws(run_root: Path = RUN_ROOT, *, verify: bool = True) -> pd.DataFrame:
    """Read ``analysis_inputs/draws.parquet``, checking its sha256 against ``manifest.json``."""
    out_dir = Path(run_root) / "analysis_inputs"
    target = out_dir / "draws.parquet"
    if verify:
        manifest = _read_json(out_dir / "manifest.json")
        if manifest["sha256"]["draws.parquet"] != _sha256(target):
            raise ValueError(f"{target}: sha256 differs from manifest.json (re-run extract)")
    return pd.read_parquet(target)
