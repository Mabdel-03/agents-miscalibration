"""Synthetic sealed worlds for the N2 (report / forecast) tests.

Builds item files in the exact ``EpisodeResult`` layout (``study/types.py``,
``tests/study/fixtures/episode_result.example.json``) and a ``SELECTIONS.json`` register in
the exact ``selection/seal.py`` layout, from hand-made candidate pools, so the compiler is
tested against the same bytes the real pipeline writes.  The VOTE records are computed with
the real ``selection.vote.vote`` so winners, ties and grouping modes are the frozen ones.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study import types as T
from agents_scaling.study.config import StudyConfig
from agents_scaling.study.parse.candidate import candidate_sha256
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.selection.seal import rekey
from agents_scaling.study.selection.vote import vote
from tests.study.conftest import StubTokenizer  # noqa: F401  (re-exported for the tests)
from tests.study.wp5_support import make_cell, make_tasks, write_export  # noqa: F401

SEAL = "ab" * 32
INVALID_SHA = identity.sha256_hex(T.SENTINEL_JSON)
FRAMING = {T.Method.IND_VOTE: T.Framing.F11, T.Method.DEC: T.Framing.NATIVE, T.Method.CEN_FLAT: T.Framing.NATIVE}


def make_candidate(answer: str, confidence: float = 0.7, *, words: int = 0, tag: str = "") -> T.Candidate:
    """A complete candidate; ``words`` pads ``approach`` with that many whitespace tokens."""
    approach = f"approach {tag}".strip() + (" " + " ".join(f"w{tag}{i}" for i in range(words)) if words else "")
    return T.Candidate(
        approach=approach,
        evidence=(T.Evidence(f"claim {tag}", f"support {tag}", "low"),),
        alternatives_considered=(f"alt {tag}",),
        failure_checks=(f"check {tag}",),
        final_answer=answer,
        confidence=confidence,
    )


def make_record(seed: str, slot: int, stage: str, candidate: T.Candidate | None, *, failure_code: str = "SCHEMA") -> T.CandidateRecord:
    request_id = identity.sha256_hex(f"request:{seed}")
    valid = candidate is not None
    return T.CandidateRecord(
        candidate_id=identity.sha256_hex(request_id),
        request_id=request_id,
        slot=slot,
        stage=stage,
        valid=valid,
        failure_code=None if valid else failure_code,
        candidate=candidate,
        candidate_sha256=candidate_sha256(candidate) if valid else INVALID_SHA,
        raw_content_sha256=identity.sha256_hex(f"raw:{seed}"),
    )


def ledger_summary(B_flops: int = 4_000_000, spent: int = 3_100_000, calls: int = 6, stop: str = "CALL_CAP") -> dict[str, Any]:
    return {
        "B_flops": B_flops, "spent": spent, "reserved_open": 0, "peak_reserved": spent, "peak_allocated": spent,
        "slack": B_flops - spent, "final_reserve": 0, "final_reserve_calls": 0, "final_reserve_released": False,
        "calls_admitted": calls, "selector_calls_admitted": 0, "solver_call_cap": 64, "selector_call_cap": 64,
        "stop_reason": stop, "stop_detail": {}, "closed": True, "oracle": "0" * 64, "events": [],
    }


def counters_block(records: Sequence[T.CandidateRecord], N: int, roles: dict[str, int], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "assigned_roster": N, "unique_actors_used": N, "context_epochs": sum(roles.values()), "resets": 0,
        "peak_live_contexts": N, "model_invocations": sum(roles.values()), "opportunities": sum(roles.values()),
        "calls_by_role": dict(roles), "context_failures_by_role": {}, "tokens_by_role_channel": {},
        "complete_candidates": sum(1 for r in records if r.valid), "candidate_opportunities": len(records),
        "aliased_calls": 0, "generated_calls": sum(roles.values()), "pregenerated_unadmitted": 0,
    }
    if extra:
        out.update(extra)
    return out


def write_item(
    run_root: Path,
    cell: T.CellSpec,
    task: T.PublicTask,
    records: Sequence[T.CandidateRecord],
    *,
    study_seed: bytes,
    pool_ids: Sequence[str] | None = None,
    ledger: dict[str, Any] | None = None,
    counters_extra: dict[str, Any] | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    """Write ``cells/<cell>/items/<sid>.json`` with the method's own selection block."""
    method = T.Method(cell.method)
    records = rekey(list(records), task.answer_format)
    by_id = {r.candidate_id: r for r in records}
    pool = [by_id[c] for c in pool_ids] if pool_ids is not None else list(records)
    ledger = ledger or ledger_summary()
    if method is T.Method.CEN_FLAT:
        selection = None
        native = records[-1]
        roles = {"hub": 1 + counters_extra.get("delegation_cycles", 0) if counters_extra else 1, "worker": 4 * (counters_extra or {}).get("delegation_cycles", 0)}
    else:
        rec = vote(pool, task.source_id, study_seed, task.answer_format, pool_kind=seals.POOL_ARCHIVE if method is T.Method.IND_VOTE else seals.POOL_LATEST)
        selection = rec.to_dict()
        native = by_id.get(rec.selected_candidate_id) if rec.selected_candidate_id else None
        roles = {"root": len(records)} if method is T.Method.IND_VOTE else {"root": cell.N, "revise": len(records) - cell.N}
    episode = {
        "episode_id": identity.sha256_hex(identity.jcs([cell.cell_id, task.source_id])),
        "method": method.value, "N": cell.N, "B": cell.B, "framing": cell.framing.value,
        "calls": [], "candidates": [r.to_dict() for r in records], "packets": [],
        "ledger": ledger, "counters": counters_block(records, cell.N, roles, counters_extra),
        "selection": selection,
        "native_final": None if native is None else {
            "candidate_id": native.candidate_id, "valid": native.valid,
            "final_answer": None if native.candidate is None else native.candidate.final_answer,
            "confidence": None if native.candidate is None else native.candidate.confidence,
        },
        "stop_reason": ledger["stop_reason"],
    }
    if method is T.Method.DEC:
        episode["latest_candidate_ids"] = [r.candidate_id for r in pool]
    result = T.EpisodeResult(1, cell, task.source_id, T.Domain(task.domain), task.split, status, episode, None,
                             {"started_at": 1.0, "finished_at": 2.0, "wall_s": 1.0}, {"slurm_job_id": None, "host": "test"}, "test")
    data = result.to_dict()
    io.write_json(seals.item_path(run_root, cell.cell_id, task.source_id), data)
    return data


def seal_items(run_root: Path, cfg: StudyConfig, cells: Sequence[T.CellSpec], tasks: Sequence[T.PublicTask], *, seal: str = SEAL,
               skip: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """Write ``seals/<seal>/SELECTIONS.json`` (and POOLS.json) from the written item files
    with the real pool builder and VOTE; ``skip`` = (cell_id, source_id) pairs left unsealed."""
    formats = {t.source_id: t.answer_format for t in tasks}
    pools: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for cell in cells:
        for sid in cell.items:
            if (cell.cell_id, sid) in skip:
                continue
            item = seals.load_item_file(run_root, cell.cell_id, sid)
            if item is None:
                continue
            for pool in seals.pools_for_item(cell, item, formats[sid]):
                pools[pool["pool_id"]] = pool
                records = rekey([r for r in seals.candidate_records_of(item) if r.candidate_id in set(pool["candidate_ids"])], formats[sid])
                by_id = {r.candidate_id: r for r in records}
                ordered = [by_id[c] for c in pool["candidate_ids"]]
                rec = vote(ordered, sid, cfg.study_seed, formats[sid], pool_kind=pool["pool_kind"])
                entry = {
                    "selection_id": identity.sha256_hex(identity.jcs([pool["pool_id"], rec.selector_id])),
                    "pool_id": pool["pool_id"], "selector_id": rec.selector_id, "cell_id": pool["cell_id"], "source_id": sid,
                    "domain": pool["domain"], "module": pool["module"], "method": pool["method"], "checkpoint": pool["checkpoint"],
                    "N": pool["N"], "B": pool["B"], "framing": pool["framing"], "episode_rep": pool["episode_rep"], "degree": pool["degree"],
                    "pool_kind": pool["pool_kind"], "prefix_k": pool["prefix_k"], "record": rec.to_dict(), "score_cell_id": None,
                    "sealed_at": 1000.0, "sealed_at_iso": "2026-09-05T00:00:00+00:00",
                }
                selections[entry["selection_id"]] = entry
    root = seals.seals_dir(run_root, seal)
    root.mkdir(parents=True, exist_ok=True)
    (root / seals.POOLS_FILE).write_text(json.dumps({"schema_version": 1, "seal": seal, "pools": pools}, indent=1, sort_keys=True))
    register = {"schema_version": 1, "seal": seal, "selections": selections, "n_selections": len(selections)}
    (root / seals.SELECTIONS_FILE).write_text(json.dumps(register, indent=1, sort_keys=True))
    return register


def cell_for(method: T.Method, items: Sequence[str], **kwargs: Any) -> T.CellSpec:
    return make_cell(method, items, module="A", N=5, B=4, framing=FRAMING[method], **kwargs)


class World:
    """A run root with a public export, three A cells on the items and the item files written
    from the pools returned by ``pool_fn(method, task) -> (records, pool_ids | None, counters_extra)``."""

    def __init__(self, run_root: Path, cfg: StudyConfig, tasks: Sequence[T.PublicTask]) -> None:
        self.run_root = run_root
        self.cfg = cfg
        self.tasks = list(tasks)
        self.by_id = {t.source_id: t for t in tasks}
        self.cells: dict[T.Method, T.CellSpec] = {}
        self.items: dict[tuple[T.Method, str], dict[str, Any]] = {}
        for name in ("servers", "cells", "requests", "seals", "forecast"):
            (run_root / name).mkdir(parents=True, exist_ok=True)
        write_export(run_root, tasks)

    def add(self, method: T.Method, pools: dict[str, tuple[Sequence[T.CandidateRecord], Sequence[str] | None, dict[str, Any] | None]],
            ledger: dict[str, Any] | None = None) -> T.CellSpec:
        cell = cell_for(method, [t.source_id for t in self.tasks])
        self.cells[method] = cell
        for sid, (records, pool_ids, extra) in pools.items():
            self.items[(method, sid)] = write_item(self.run_root, cell, self.by_id[sid], records, study_seed=self.cfg.study_seed,
                                                   pool_ids=pool_ids, counters_extra=extra, ledger=ledger)
        return cell

    def seal(self, **kwargs: Any) -> dict[str, Any]:
        return seal_items(self.run_root, self.cfg, list(self.cells.values()), self.tasks, **kwargs)


def ind_pool(tag: str, answers: Sequence[str], *, words: int = 0, invalid: Sequence[int] = ()) -> list[T.CandidateRecord]:
    out = []
    for i, ans in enumerate(answers):
        cand = None if i in invalid else make_candidate(ans, 0.5 + 0.05 * i, words=words, tag=f"{tag}{i}")
        out.append(make_record(f"{tag}:{i}", i % 5, "root", cand))
    return out


__all__ = [name for name in globals() if not name.startswith("_")]
