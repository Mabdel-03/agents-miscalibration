"""Capture CLI: resumable shard runner over stored requests / compiled reports (N1).

    python -m agents_scaling.study.neural.capture --run-id study_v4 --stage native \\
        --checkpoint 32B --blocks 15,31,47 --shard 0 --num-shards 12 [--max-batch-tokens 16384] [--fidelity 20]
    python -m agents_scaling.study.neural.capture --run-id study_v4 --stage report --checkpoint 32B --shard 0 --num-shards 4

Run with the GPU python (``/orcd/data/tpoggio/001/mabdel03/envs/neural_env/bin/python``);
``agents_scaling`` is not installed there, so this module puts ``<repo>/src`` on
``sys.path`` itself (``PYTHONPATH=src`` is only needed for the ``-m`` form).

Stage ``native`` (brief R2/R3): the flagship N=5/B4 policies on the panel (``panel.py``:
``PublicTask.rank < 150`` per superdomain on ``main``), one episode each; per episode one
call per declared role/phase chosen by an outcome-blind HMAC over the request ids, except
the *geometry groups* (``GEOMETRY_GROUPS``: IND roots, DEC terminal members, CEN_FLAT hub /
worker roots) where every eligible call is kept with its HMAC ``group_rank`` so R3 has its
s = 5 comparable states (``select_native_calls``; the rule and every group size are written
to ``<run_root>/neural/native/selection.<shard>.json``).  Each selected call is
teacher-forced from its stored ``prompt_token_ids + token_ids[:k_max]`` and captured at
NATIVE_PREFILL, GENERATED_32/128/512 and FINAL_OBJECT_CLOSE.

Stage ``report`` (brief R1): the compiled FINAL_HANDOFF_REPORT renders produced by N2
(``forecast.manifest.report_render``) under ``<run_root>/forecast/reports/<item>.<method>.json``
(interface: ``report_id, source_id, method, messages, prompt_token_ids, chat_template_kwargs,
checkpoint.model_revision``, anchors as ``byte_anchors``/``anchor_tokens`` and/or the flat
``task_only_anchor_byte, state_anchor_byte, task_only_anchor_token, state_anchor_token``),
captured at TASK_ONLY_ANCHOR, STATE_ANCHOR and LAST_PREFILL.  A render whose task-only
anchor cannot be resolved, whose state anchor would fall back to the last prefill token, or
whose ``checkpoint.model_revision`` is not the engine snapshot is refused (P0-1, P1-3).

Both stages: exact token identity is asserted before any forward pass; rows are written
as ``ActivationRow`` ledger lines with fp16 vectors (``storage``); keys already committed
in the stage directory are skipped, so a re-submitted shard resumes.  The capture never
reads ``data/protected`` or any correctness label.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:  # neural_env has no agents_scaling install
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from agents_scaling.study import identity  # noqa: E402
from agents_scaling.study.neural import anchors as A  # noqa: E402
from agents_scaling.study.neural import storage as S  # noqa: E402
from agents_scaling.study.neural.engine import (  # noqa: E402
    DEFAULT_MAX_BATCH_TOKENS,
    DEFAULT_MAX_SEQ_TOKENS,
    HOOK_CONVENTION,
    CaptureRequest,
    blocks_for,
    snapshot_dir,
)
from agents_scaling.study.neural.replay import check_prompt_identity, fidelity, summarize_fidelity, teacher_forced_sequence  # noqa: E402

DEFAULT_HF_HOME = "/orcd/data/tpoggio/001/mabdel03/.cache/huggingface"
DEFAULT_RESULTS_ROOT = "/orcd/data/tpoggio/001/mabdel03/agents_scaling_results"
SELECTION_NAMESPACE = "NEURAL_R2"
R2_METHODS: tuple[str, ...] = ("IND_VOTE", "DEC", "CEN_FLAT")
R2_MODULES: tuple[str, ...] = ("A", "N")
PHASES: tuple[str, ...] = ("ROOT", "COORDINATION", "TERMINAL")
REPORTS_DIR = ("forecast", "reports")

#: Declared roles of the brief (R2) keyed by (method, native call role).
DECLARED_ROLES: Mapping[tuple[str, str], str] = {
    ("IND_VOTE", "root"): "INDEPENDENT_SOLVER",
    ("DEC", "root"): "DECENTRALIZED_MEMBER",
    ("DEC", "revise"): "DECENTRALIZED_MEMBER",
    ("CEN_FLAT", "hub"): "CENTRAL_HUB",
    ("CEN_FLAT", "worker"): "CENTRAL_WORKER",
}

#: Groups whose every eligible call is captured (brief R3: "IND: 5 roots; DEC: 5 terminal
#: members; CEN_FLAT: hub + up to 4 workers, reported separately as different roles").  The
#: HMAC still orders the group (``group_rank`` 0 is the call the one-per-group rule picks), so
#: inclusion stays outcome-blind; every other (role, phase) keeps exactly one call (P1-1).
GEOMETRY_GROUPS: frozenset[tuple[str, str]] = frozenset({
    ("INDEPENDENT_SOLVER", "ROOT"),
    ("DECENTRALIZED_MEMBER", "TERMINAL"),
    ("CENTRAL_HUB", "ROOT"),
    ("CENTRAL_WORKER", "ROOT"),
})

SELECTION_RULE = (
    "per (episode, declared_role, phase): among calls with a committed request record "
    "(no context failure), order by HMAC_sha256(study_seed, JCS(['NEURAL_R2', episode_id, "
    "declared_role, phase, request_id])); keep the argmin (group_rank 0) — except the geometry "
    "groups (INDEPENDENT_SOLVER/ROOT, DECENTRALIZED_MEMBER/TERMINAL, CENTRAL_HUB/ROOT, "
    "CENTRAL_WORKER/ROOT), where every eligible call is kept with its HMAC group_rank so R3 has "
    "its s comparable states; phases: IND_VOTE root->ROOT; DEC root->ROOT, revise "
    "step<max->COORDINATION, revise step==max->TERMINAL; CEN_FLAT hub step0->ROOT, last hub call "
    "(if not step0)->TERMINAL, other hub->COORDINATION, worker step0->ROOT, other worker->COORDINATION. "
    "Independent of call length, validity, correctness and activations."
)


def log(msg: str) -> None:
    print(f"[neural {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- work items


@dataclass(frozen=True)
class NativeWorkItem:
    request_id: str
    source_id: str
    cell_id: str
    episode_id: str
    method: str
    native_role: str
    declared_role: str
    phase: str
    actor_slot: int | None
    step: int | None
    owner: str | None
    group_size: int
    group_context_failures: int
    selection_key: str
    group_rank: int = 0  # HMAC order inside the (episode, role, phase) group; 0 = the one-per-group pick

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ReportWorkItem:
    report_id: str
    path: str
    source_id: str | None
    method: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def phase_of(method: str, call: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> str | None:
    """The named ROOT/COORDINATION/TERMINAL phase of one ``episode.calls[]`` entry (or None)."""
    role = call.get("role")
    step = int(call.get("step") or 0)
    if method == "IND_VOTE":
        return "ROOT" if role == "root" else None
    if method == "DEC":
        if role == "root":
            return "ROOT"
        if role == "revise":
            max_step = max(int(c.get("step") or 0) for c in calls if c.get("role") == "revise")
            return "TERMINAL" if step == max_step else "COORDINATION"
        return None
    if method == "CEN_FLAT":
        if role == "hub":
            if step == 0:
                return "ROOT"
            hub_indices = [i for i, c in enumerate(calls) if c.get("role") == "hub"]
            return "TERMINAL" if calls.index(call) == hub_indices[-1] else "COORDINATION"
        if role == "worker":
            return "ROOT" if step == 0 else "COORDINATION"
        return None
    return "ROOT" if role == "root" else None


def select_native_calls(study_seed: bytes, method: str, cell_id: str, source_id: str, episode: Mapping[str, Any], *,
                        geometry_groups: frozenset[tuple[str, str]] = GEOMETRY_GROUPS) -> tuple[list[NativeWorkItem], list[dict[str, Any]]]:
    """Outcome-blind selection of the captured calls of one episode: one call per (declared
    role, phase), every eligible call for the ``geometry_groups`` (ordered by the same HMAC,
    ``group_rank`` 0 being the one-per-group pick).

    Returns the selected work items and the per-group bookkeeping (eligible counts, context
    failures → the role-specific inclusion probabilities the spec asks to keep, and the
    captured request ids).
    """
    calls = list(episode.get("calls") or [])
    episode_id = str(episode.get("episode_id") or identity.sha256_hex(identity.jcs([cell_id, source_id])))
    groups: dict[tuple[str, str], list[tuple[bytes, Mapping[str, Any]]]] = {}
    failures: dict[tuple[str, str], int] = {}
    for call in calls:
        role = call.get("role")
        declared = DECLARED_ROLES.get((method, role))
        if declared is None:
            continue
        phase = phase_of(method, call, calls)
        if phase is None:
            continue
        key = (declared, phase)
        if call.get("context_failure") or not call.get("request_id"):
            failures[key] = failures.get(key, 0) + 1
            groups.setdefault(key, [])
            continue
        hkey = identity.blind_order_key(study_seed, SELECTION_NAMESPACE, episode_id, declared, phase, call["request_id"])
        groups.setdefault(key, []).append((hkey, call))
    items: list[NativeWorkItem] = []
    bookkeeping: list[dict[str, Any]] = []
    for (declared, phase) in sorted(groups):
        eligible = sorted(groups[(declared, phase)], key=lambda kv: kv[0])
        geometry = (declared, phase) in geometry_groups
        entry = {
            "episode_id": episode_id, "cell_id": cell_id, "source_id": source_id, "method": method,
            "declared_role": declared, "phase": phase, "eligible": len(eligible),
            "context_failures": failures.get((declared, phase), 0), "selected": None,
            "geometry_group": geometry, "captured": [],
        }
        keep = eligible if geometry else eligible[:1]
        for rank, (hkey, call) in enumerate(keep):
            item = NativeWorkItem(
                request_id=str(call["request_id"]), source_id=source_id, cell_id=cell_id, episode_id=episode_id,
                method=method, native_role=str(call.get("role")), declared_role=declared, phase=phase,
                actor_slot=call.get("actor_slot"), step=call.get("step"), owner=call.get("owner"),
                group_size=len(eligible), group_context_failures=failures.get((declared, phase), 0), selection_key=hkey.hex(),
                group_rank=rank,
            )
            items.append(item)
            entry["captured"].append(item.request_id)
        if keep:
            entry["selected"] = entry["captured"][0]
        bookkeeping.append(entry)
    return items, bookkeeping


def _cell_shard_index(cell_id: str) -> int:
    tail = cell_id.rsplit(".s", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def panel_cells(cells: Iterable[Any], *, checkpoint: str, N: int, B: int, methods: Sequence[str], modules: Sequence[str], split: str = "main", episode_rep: int = 0) -> list[Any]:
    """The generate cells of the R2 panel, ordered by module priority then shard index."""
    out = []
    for cell in cells:
        kind = getattr(cell.kind, "value", cell.kind)
        method = getattr(cell.method, "value", cell.method)
        if kind != "GENERATE" or method not in methods or cell.module not in modules:
            continue
        if cell.checkpoint != checkpoint or int(cell.N) != int(N) or int(cell.B) != int(B):
            continue
        if cell.split != split or int(cell.episode_rep) != int(episode_rep) or getattr(cell, "degree", None) is not None:
            continue
        out.append(cell)
    return sorted(out, key=lambda c: (modules.index(c.module), _cell_shard_index(c.cell_id)))


def native_work_list(run_root: Path, cfg: Any, cells: Sequence[Any], *, checkpoint: str, panel_items: int, methods: Sequence[str], modules: Sequence[str],
                     allow_panel_fallback: bool = True) -> tuple[list[NativeWorkItem], dict[str, Any]]:
    """Every selected call of the panel (``panel_items`` items = ``panel_items // 2`` per
    superdomain), one episode per (method, item); incomplete items are reported.

    The panel is the rank prefix of the public task export (``panel.panel_source_ids``:
    ``PublicTask.rank < per_domain`` on ``main`` — the same rule N2 and N3 apply, P1-4).
    Only when the export is absent (unit tests, ad-hoc runs) and ``allow_panel_fallback``
    does it fall back to counting the first ``per_domain`` items per superdomain in
    cell/rank order; ``meta["panel_rule"]`` records which rule produced the list and
    ``meta["panel"]`` the resolved item ids.
    """
    from agents_scaling.study.neural.panel import PANEL_RULE, panel_source_ids
    from agents_scaling.study.selection.seal import load_item_file

    study_seed = cfg.study_seed
    B4 = int(cfg.budget.primary)
    chosen = panel_cells(cells, checkpoint=checkpoint, N=5, B=B4, methods=methods, modules=modules)
    per_domain = max(1, panel_items // 2)
    try:
        panel: dict[str, list[str]] | None = panel_source_ids(run_root, per_domain)
        panel_rule = PANEL_RULE
    except FileNotFoundError:
        if not allow_panel_fallback:
            raise
        panel, panel_rule = None, "cell_order_fallback: first per_domain items per superdomain in cell/rank order (no data/public/tasks.jsonl)"
        log("WARNING: no public task export; panel falls back to cell order (not the rank rule)")
    allowed = None if panel is None else {sid for ids in panel.values() for sid in ids}
    items: list[NativeWorkItem] = []
    groups: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts = {"cells": len(chosen), "episodes": 0, "missing_item_files": 0, "incomplete": 0, "skipped_duplicates": 0, "outside_panel": 0}
    domain_counts: dict[str, dict[str, int]] = {}
    for cell in chosen:
        method = getattr(cell.method, "value", cell.method)
        dc = domain_counts.setdefault(method, {})
        for source_id in cell.items:
            domain = source_id.split(":", 1)[0]
            if (method, source_id) in seen:
                counts["skipped_duplicates"] += 1
                continue
            if allowed is not None:
                if source_id not in allowed:
                    counts["outside_panel"] += 1
                    continue
            elif dc.get(domain, 0) >= per_domain:
                continue
            item = load_item_file(run_root, cell.cell_id, source_id)
            if item is None:
                counts["missing_item_files"] += 1
                continue
            if item.get("status") != "complete" or not isinstance(item.get("episode"), dict):
                counts["incomplete"] += 1
                continue
            seen.add((method, source_id))
            dc[domain] = dc.get(domain, 0) + 1
            counts["episodes"] += 1
            selected, bookkeeping = select_native_calls(study_seed, method, cell.cell_id, source_id, item["episode"])
            items.extend(selected)
            groups.extend(bookkeeping)
    resolved: dict[str, dict[str, list[str]]] = {}
    for w in items:
        lst = resolved.setdefault(w.method, {}).setdefault(w.source_id.split(":", 1)[0], [])
        if w.source_id not in lst:
            lst.append(w.source_id)
    meta = {
        "rule": SELECTION_RULE, "namespace": SELECTION_NAMESPACE, "geometry_groups": sorted(list(g) for g in GEOMETRY_GROUPS),
        "methods": list(methods), "modules": list(modules),
        "checkpoint": checkpoint, "N": 5, "B": B4, "panel_items": panel_items, "per_domain": per_domain,
        "panel_rule": panel_rule, "panel": panel, "panel_resolved": resolved,
        "cells": [c.cell_id for c in chosen], "counts": counts, "items_per_method_domain": domain_counts,
        "groups": groups, "work_sha256": identity.sha256_hex(json.dumps([w.to_dict() for w in items], sort_keys=True)),
    }
    return items, meta


def load_items_file(path: Path, stage: str) -> list[Any]:
    """Explicit work list: JSON list or JSONL of ``NativeWorkItem``/``ReportWorkItem`` dicts
    (native rows need at least ``request_id``; report rows at least ``path``)."""
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    out: list[Any] = []
    for i, row in enumerate(rows):
        if stage == "native":
            rid = row["request_id"] if isinstance(row, Mapping) else str(row)
            base = dict(request_id=rid, source_id="?", cell_id="?", episode_id="?", method="?", native_role="root",
                        declared_role="?", phase="?", actor_slot=None, step=None, owner=None, group_size=1,
                        group_context_failures=0, selection_key="", group_rank=0)
            if isinstance(row, Mapping):
                base.update({k: v for k, v in row.items() if k in base})
            out.append(NativeWorkItem(**base))
        else:
            p = row["path"] if isinstance(row, Mapping) else str(row)
            rid = row.get("report_id") if isinstance(row, Mapping) else None
            out.append(ReportWorkItem(report_id=rid or Path(p).stem, path=str(p), source_id=row.get("source_id") if isinstance(row, Mapping) else None,
                                      method=row.get("method") if isinstance(row, Mapping) else None))
    return out


def report_work_list(run_root: Path) -> list[ReportWorkItem]:
    directory = run_root.joinpath(*REPORTS_DIR)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            head = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"{path}: not valid JSON ({exc})") from exc
        out.append(ReportWorkItem(report_id=str(head.get("report_id") or path.stem), path=str(path),
                                  source_id=head.get("source_id"), method=head.get("method")))
    return out


def shard_of(items: Sequence[Any], shard: int, num_shards: int) -> list[Any]:
    if num_shards < 1 or not (0 <= shard < num_shards):
        raise ValueError(f"shard {shard} outside [0, {num_shards})")
    return [w for i, w in enumerate(items) if i % num_shards == shard]


# --------------------------------------------------------------------------- rows


def _row(
    *, stage: str, snapshot_id: str, revision: str, condition: str, nonce_hash: str, block: int, anchor: A.Anchor,
    result: Any, hidden_size: int, checkpoint: str, extra: Mapping[str, Any], **join: Any,
) -> tuple[S.ActivationRow, np.ndarray | None]:
    vector = result.residual_at(block, anchor.token_offset) if anchor.present and result is not None else None
    row = S.ActivationRow(
        StateSnapshot_id=snapshot_id, consumer_revision=revision, condition=condition, child_slot=None,
        nonce_hash=nonce_hash, block=int(block), anchor_kind=anchor.kind, structural_span=anchor.structural_span,
        token_offset=anchor.token_offset, channel=anchor.channel, generated_token_count=int(anchor.generated_token_count),
        missingness=anchor.missingness, tensor_hash=None,
        measurement_cost=result.measurement_cost if result is not None else {"sequence_tokens": 0, "share_seconds": 0.0},
        operationally_available_at_checkpoint=True, stage=stage, sequence_id=snapshot_id,
        hidden_size=hidden_size, sequence_tokens=None if result is None else int(result.n_tokens),
        hook_convention=HOOK_CONVENTION, checkpoint=checkpoint, extra={**dict(extra), "anchor_detail": anchor.detail}, **join,
    )
    return row, vector


def _pending(items: Sequence[Any], done: set[tuple[str, int, str]], blocks: Sequence[int], kinds: Sequence[str], id_of) -> list[Any]:
    out = []
    for w in items:
        sid = id_of(w)
        if all((sid, int(b), k) in done for b in blocks for k in kinds):
            continue
        out.append(w)
    return out


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# --------------------------------------------------------------------------- runners


def run_native(args: argparse.Namespace, engine: Any, tokenizer: Any, run_root: Path, items: Sequence[NativeWorkItem], revision: str, checkpoint: Any) -> dict[str, Any]:
    from agents_scaling.study.inference.store import RequestStore

    store = RequestStore(run_root / "requests")
    blocks = list(engine.blocks)
    kinds = [A.ANCHOR_NATIVE_PREFILL, *[A.generated_anchor_kind(k) for k in args.generated_ks], A.ANCHOR_FINAL_OBJECT_CLOSE]
    done = S.existing_keys(run_root, "native")
    pending = _pending(items, done, blocks, kinds, lambda w: w.request_id)
    log(f"native shard {args.shard}/{args.num_shards}: {len(items)} work items, {len(pending)} pending ({len(done)} keys already stored)")
    stats: dict[str, Any] = {"work_items": len(items), "pending": len(pending), "captured": 0, "missing_records": 0, "rows": 0, "tokens": 0, "anchors_missing": {}}
    fidelity_reports: list[dict[str, Any]] = []
    with S.ShardWriter(run_root, "native", args.shard_name, chunk_rows=args.chunk_rows) as writer:
        for group in _chunks(pending, args.group_size):
            requests, meta = [], []
            for w in group:
                record = store.get(w.request_id)
                if record is None:
                    stats["missing_records"] += 1
                    if args.strict_records:
                        raise RuntimeError(f"request record {w.request_id} not in {store.root}")
                    log(f"missing record {w.request_id[:12]} (skipped)")
                    continue
                model_rev = str(record.identity.get("model_revision", ""))
                if model_rev != revision and not args.allow_model_mismatch:
                    raise RuntimeError(f"{w.request_id[:12]}: record model_revision {model_rev[:8]} != engine {revision[:8]} (pass --allow-model-mismatch to override)")
                check_prompt_identity(record, tokenizer, strict=True)
                anchors = A.resolve_native_anchors(record, tokenizer, final_object=A.final_object_for_role(w.native_role), ks=args.generated_ks)
                k_max = A.k_max_needed(anchors)
                seq = teacher_forced_sequence(record, k_max)
                positions = tuple(sorted({a.token_offset for a in anchors if a.present}))
                requests.append(CaptureRequest(w.request_id, seq, positions))
                meta.append((w, record, anchors, k_max, model_rev))
                if args.fidelity and len(fidelity_reports) < args.fidelity and record.response.get("token_ids"):
                    fidelity_reports.append(fidelity(record, engine, args.fidelity_tokens, tokenizer=tokenizer))
            if not requests:
                continue
            results = engine.forward(requests)
            for (w, record, anchors, k_max, model_rev), result in zip(meta, results):
                nonce = str(record.identity.get("input_hash") or identity.input_hash(record.messages, record.chat_template_kwargs))
                extra = {
                    "native_role": w.native_role, "actor_slot": w.actor_slot, "step": w.step, "owner": w.owner,
                    "group_size": w.group_size, "group_context_failures": w.group_context_failures, "selection_key": w.selection_key,
                    "group_rank": w.group_rank, "k_max": k_max, "prompt_tokens": len(record.prompt_token_ids), "completion_tokens": len(record.response.get("token_ids") or []),
                    "finish_reason": record.response.get("finish_reason"), "record_model_revision": model_rev, "engine_seed": record.engine_seed,
                }
                for anchor in anchors:
                    if not anchor.present:
                        stats["anchors_missing"][anchor.missingness] = stats["anchors_missing"].get(anchor.missingness, 0) + 1
                    for b in blocks:
                        row, vec = _row(stage="native", snapshot_id=w.request_id, revision=revision, condition=f"native:{w.method}",
                                        nonce_hash=nonce, block=b, anchor=anchor, result=result, hidden_size=engine.hidden_size,
                                        checkpoint=checkpoint.size, extra=extra, source_id=w.source_id, method=w.method,
                                        role=w.declared_role, phase=w.phase, cell_id=w.cell_id, episode_id=w.episode_id)
                        writer.add(row, vec)
                        stats["rows"] += 1
                stats["captured"] += 1
                stats["tokens"] += result.n_tokens
            writer.flush()
            log(f"captured {stats['captured']}/{len(pending)} sequences, {stats['tokens']} tokens, {engine.tokens_per_second:.0f} tok/s")
    if fidelity_reports:
        S.atomic_write_json(S.stage_dir(run_root, "native") / f"fidelity.{args.shard_name}.json",
                            {"summary": summarize_fidelity(fidelity_reports), "tokens_per_record": args.fidelity_tokens, "records": fidelity_reports})
        stats["fidelity"] = summarize_fidelity(fidelity_reports)
    return stats


def load_report(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in A.REPORT_REQUIRED_FIELDS if k not in data]
    if missing:
        raise RuntimeError(f"{path}: report lacks {missing}")
    return data


def check_report_anchors(report_id: str, anchors: Sequence[A.Anchor]) -> None:
    """P0-1: the N2 → N1 anchor interface is required.  Refuse a render whose task-only
    anchor is unresolved or whose state anchor is the last-prefill default (that would
    silently label the last chat-template token as STATE_ANCHOR)."""
    by = {a.kind: a for a in anchors}
    task, state = by[A.ANCHOR_TASK_ONLY], by[A.ANCHOR_STATE]
    if not task.present:
        raise RuntimeError(f"{report_id}: TASK_ONLY_ANCHOR unresolved (source={task.detail.get('source')}); the render must carry "
                           "byte_anchors/anchor_tokens (forecast.manifest.report_render) or task_only_anchor_byte/_token")
    if state.detail.get("source") == "last_prefill_default":
        raise RuntimeError(f"{report_id}: STATE_ANCHOR absent from the render (would default to the last prefill token); "
                           "the render must carry byte_anchors/anchor_tokens or state_anchor_byte/_token")


def report_model_revision(report: Mapping[str, Any]) -> str:
    ckpt = report.get("checkpoint")
    if isinstance(ckpt, Mapping) and ckpt.get("model_revision"):
        return str(ckpt["model_revision"])
    return str(report.get("model_revision") or "")


def run_report(args: argparse.Namespace, engine: Any, tokenizer: Any, run_root: Path, items: Sequence[ReportWorkItem], revision: str, checkpoint: Any) -> dict[str, Any]:
    blocks = list(engine.blocks)
    kinds = [A.ANCHOR_TASK_ONLY, A.ANCHOR_STATE, A.ANCHOR_LAST_PREFILL]
    done = S.existing_keys(run_root, "report")
    pending = _pending(items, done, blocks, kinds, lambda w: w.report_id)
    log(f"report shard {args.shard}/{args.num_shards}: {len(items)} reports, {len(pending)} pending ({len(done)} keys already stored)")
    stats: dict[str, Any] = {"work_items": len(items), "pending": len(pending), "captured": 0, "rows": 0, "tokens": 0, "anchors_missing": {}}
    with S.ShardWriter(run_root, "report", args.shard_name, chunk_rows=args.chunk_rows) as writer:
        for group in _chunks(pending, args.group_size):
            requests, meta = [], []
            for w in group:
                report = load_report(Path(w.path))
                render_rev = report_model_revision(report)
                if render_rev != revision and not args.allow_model_mismatch:
                    raise RuntimeError(f"{w.report_id}: render checkpoint.model_revision {render_rev[:8] or '<absent>'} != engine {revision[:8]} "
                                       "(pass --allow-model-mismatch to override)")
                ids = tuple(int(t) for t in report["prompt_token_ids"])
                ctk = report.get("chat_template_kwargs") or {"enable_thinking": False}
                pseudo = {"request_id": w.report_id, "prompt_token_ids": ids, "messages": report["messages"], "chat_template_kwargs": ctk, "response": {"token_ids": []}}
                check_prompt_identity(pseudo, tokenizer, strict=True)
                anchors = A.resolve_report_anchors(report, tokenizer)
                check_report_anchors(w.report_id, anchors)
                positions = tuple(sorted({a.token_offset for a in anchors if a.present}))
                requests.append(CaptureRequest(w.report_id, ids, positions))
                meta.append((w, report, anchors, render_rev))
            if not requests:
                continue
            results = engine.forward(requests)
            for (w, report, anchors, render_rev), result in zip(meta, results):
                nonce = identity.input_hash(report["messages"], report.get("chat_template_kwargs") or {"enable_thinking": False})
                method = report.get("method") or w.method
                inner = report.get("report") if isinstance(report.get("report"), Mapping) else {}
                extra = {"report_path": w.path, "prompt_tokens": len(report["prompt_token_ids"]),
                         "report_sha256": report.get("report_sha256") or inner.get("text_sha256"),
                         "rendered_text_sha256": report.get("rendered_text_sha256"), "forecast_request_id": report.get("request_id"),
                         "render_model_revision": render_rev, "seal": report.get("seal"), "pool_id": report.get("pool_id"),
                         "manifest": report.get("manifest"), "selection_id": report.get("selection_id")}
                for anchor in anchors:
                    if not anchor.present:
                        stats["anchors_missing"][anchor.missingness] = stats["anchors_missing"].get(anchor.missingness, 0) + 1
                    for b in blocks:
                        row, vec = _row(stage="report", snapshot_id=w.report_id, revision=revision, condition=f"report:{method}",
                                        nonce_hash=nonce, block=b, anchor=anchor, result=result, hidden_size=engine.hidden_size,
                                        checkpoint=checkpoint.size, extra=extra, source_id=report.get("source_id") or w.source_id,
                                        method=method, role="REPORT_READER", phase="FINAL_HANDOFF_REPORT", cell_id=report.get("cell_id"),
                                        episode_id=report.get("episode_id") or (inner.get("item") or {}).get("episode_id"))
                        writer.add(row, vec)
                        stats["rows"] += 1
                stats["captured"] += 1
                stats["tokens"] += result.n_tokens
            writer.flush()
            log(f"captured {stats['captured']}/{len(pending)} reports, {stats['tokens']} tokens, {engine.tokens_per_second:.0f} tok/s")
    return stats


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agents_scaling.study.neural.capture", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", choices=S.STAGES, required=True)
    ap.add_argument("--checkpoint", default="32B", help="size key of configs/study_v4.yaml checkpoints (engine + record model)")
    ap.add_argument("--blocks", default=None, help="comma list; default floor(f*(L-1)) for f in 0.25,0.5,0.75")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--items-file", default=None, help="explicit work list (JSON/JSONL) instead of the panel selection / reports dir")
    ap.add_argument("--max-batch-tokens", type=int, default=DEFAULT_MAX_BATCH_TOKENS)
    ap.add_argument("--max-seq-tokens", type=int, default=DEFAULT_MAX_SEQ_TOKENS)
    ap.add_argument("--fidelity", type=int, default=20, help="records of this shard to run the §10.5 fidelity replay on (0 = off)")
    ap.add_argument("--fidelity-tokens", type=int, default=64)
    ap.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", DEFAULT_HF_HOME))
    ap.add_argument("--snapshot-path", default=None, help="override the HF snapshot directory")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16", help="model dtype (captured vectors are always fp16)")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--cells-file", action="append", default=[], help="cells manifest(s), relative to the run root; default cells_*_<checkpoint>.json")
    ap.add_argument("--methods", default=",".join(R2_METHODS))
    ap.add_argument("--modules", default=",".join(R2_MODULES))
    ap.add_argument("--panel-items", type=int, default=300)
    ap.add_argument("--generated-ks", default=",".join(str(k) for k in A.GENERATED_KS))
    ap.add_argument("--group-size", type=int, default=32, help="sequences resolved/flushed per engine call")
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="cap the shard's work items (tests)")
    ap.add_argument("--allow-model-mismatch", action="store_true", help="capture records/renders produced by another checkpoint (tests only)")
    ap.add_argument("--no-strict-records", dest="strict_records", action="store_false", help="skip work items whose record is missing")
    ap.add_argument("--dry-run", action="store_true", help="write the selection/work list and exit without loading the model")
    return ap


def _blocks_arg(value: str | None, num_layers: int) -> tuple[int, ...]:
    if not value:
        return blocks_for(num_layers)
    return tuple(sorted({int(x) for x in value.split(",") if x.strip()}))


def resolve_snapshot(args: argparse.Namespace, checkpoint: Any) -> Path:
    if args.snapshot_path:
        return Path(args.snapshot_path)
    return snapshot_dir(args.hf_home, checkpoint.hf_id, checkpoint.model_revision)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from agents_scaling.study.config import load_config

    cfg = load_config()
    checkpoint = cfg.checkpoints[args.checkpoint]
    run_root = Path(args.results_root) / args.run_id
    args.generated_ks = tuple(int(k) for k in args.generated_ks.split(",") if k.strip())
    args.shard_name = f"{args.stage}.s{args.shard:03d}of{args.num_shards:03d}"
    snapshot = resolve_snapshot(args, checkpoint)
    if not (snapshot / "config.json").is_file():
        raise SystemExit(f"snapshot {snapshot} is not in the offline HF cache")
    num_layers = int(json.loads((snapshot / "config.json").read_text())["num_hidden_layers"])
    blocks = _blocks_arg(args.blocks, num_layers)
    log(f"run_root={run_root} stage={args.stage} checkpoint={checkpoint.size}@{checkpoint.model_revision[:8]} L={num_layers} blocks={blocks}")

    # ---- work list ------------------------------------------------------------------
    meta: dict[str, Any] = {}
    if args.items_file:
        work = load_items_file(Path(args.items_file), args.stage)
        meta = {"items_file": args.items_file}
    elif args.stage == "native":
        from agents_scaling.study.cells import load_cells_file

        files = [run_root / f if not Path(f).is_absolute() else Path(f) for f in args.cells_file] or sorted(run_root.glob(f"cells_*_{args.checkpoint}.json"))
        cells = [c for f in files for c in load_cells_file(f)]
        if not cells:
            raise SystemExit(f"no cells manifests found ({files}); pass --cells-file or --items-file")
        work, meta = native_work_list(run_root, cfg, cells, checkpoint=args.checkpoint, panel_items=args.panel_items,
                                      methods=tuple(args.methods.split(",")), modules=tuple(args.modules.split(",")))
        meta["cells_files"] = [str(f) for f in files]
    else:
        work = report_work_list(run_root)
        meta = {"reports_dir": str(run_root.joinpath(*REPORTS_DIR))}
    mine = shard_of(work, args.shard, args.num_shards)
    if args.limit is not None:
        mine = mine[: args.limit]
    stage_root = S.stage_dir(run_root, args.stage)
    stage_root.mkdir(parents=True, exist_ok=True)
    S.atomic_write_json(stage_root / f"selection.{args.shard_name}.json", {
        "stage": args.stage, "shard": args.shard, "num_shards": args.num_shards, "checkpoint": checkpoint.size,
        "model_revision": checkpoint.model_revision, "blocks": list(blocks), "generated_ks": list(args.generated_ks),
        "hook_convention": HOOK_CONVENTION, "total_work_items": len(work), "shard_work_items": len(mine),
        "shard_items": [w.to_dict() for w in mine], **meta,
    })
    log(f"work list: {len(work)} total, {len(mine)} in this shard")
    if args.dry_run:
        return 0
    if not mine:
        log("nothing to do")
        return 0

    # ---- engine ---------------------------------------------------------------------
    import torch
    from transformers import AutoTokenizer

    from agents_scaling.study.neural.engine import CaptureEngine

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    engine = CaptureEngine(str(snapshot), blocks, device_map=args.device_map, dtype=dtype, max_batch_tokens=args.max_batch_tokens,
                           attn_implementation=args.attn_implementation, max_seq_tokens=args.max_seq_tokens,
                           pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0))
    log(f"engine ready in {engine.load_seconds:.0f}s: {engine.describe()}")
    try:
        runner = run_native if args.stage == "native" else run_report
        stats = runner(args, engine, tokenizer, run_root, mine, checkpoint.model_revision, checkpoint)
    finally:
        engine.close()
    stats.update({"tokens_per_second": engine.tokens_per_second, "forward_seconds": engine.forward_seconds, "batches": engine.batches_run,
                  "engine": engine.describe(), "shard": args.shard_name})
    S.atomic_write_json(stage_root / f"stats.{args.shard_name}.json", stats)
    log(f"done: {json.dumps({k: v for k, v in stats.items() if k != 'engine'}, default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
