"""Seals: frozen candidate pools and selections before any correctness join (WP5).

Spec §3.6 ("seal pool definitions and selected IDs before correctness joins"), §4.4/§5.3
(VOTE with multiplicity; JUDGE_BEST companion on the same bank), §5.5 (both selectors on
every framing bank at prefixes 1, 2, 3, 5, 9, 10), §10.3 (ledgers and seals are the join
keys).  Architecture §1.13; corrections P0-3 (order: ``seal --kind pools`` → JUDGE_BEST
cells → ``seal --kind selections`` → evaluation cells; ``aggregate`` refuses a join unless
``SELECTIONS`` were sealed before every evaluation record started), audit fixture T18
(evaluation refuses unsealed items).

Layout: ``<run_root>/seals/<manifest_sha>/POOLS.json`` and ``SELECTIONS.json`` where
``manifest_sha`` is the sha256 of the sealed generate cells file.  Both files are
append-only registers keyed by ``pool_id`` / ``selection_id``: re-running a seal adds the
newly completed items and re-asserts (never edits) the existing entries, each of which
carries its own ``sealed_at``.  Nothing here reads protected data.

Pool kinds: ``bank_prefix`` (F banks, k ∈ PREFIXES), the policy's own primary pool
(``archive`` for S_FRESH/S_HISTORY/IND_VOTE, ``latest_slots`` for DEC-family), plus a
diagnostic ``archive`` pool for DEC-family cells, and ``native`` (the single native final of
CEN_FLAT / DEGREE).  VOTE is recomputed from the sealed candidates with WP3's frozen rule
and must reproduce the policy's own recorded selection (else ``ProtocolError``).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agents_scaling.config import DEFAULT_RESULTS_ROOT
from agents_scaling.experiment import io
from agents_scaling.study import identity
from agents_scaling.study.cells import cell_seal, cells_file_sha256, load_cells_file
from agents_scaling.study.config import StudyConfig, load_config
from agents_scaling.study.selection.judge_best import judge_best_select
from agents_scaling.study.selection.normalize import vote_key
from agents_scaling.study.selection.vote import keyed_pool, prefix_vote, vote
from agents_scaling.study.types import (
    CandidateRecord,
    CellKind,
    CellSpec,
    Method,
    ProtocolError,
    SelectionRecord,
)

SEALS_DIR = "seals"
POOLS_FILE = "POOLS.json"
SELECTIONS_FILE = "SELECTIONS.json"
PREFIXES: tuple[int, ...] = (1, 2, 3, 5, 9, 10)
POOL_BANK_PREFIX = "bank_prefix"
POOL_NATIVE = "native"
POOL_ARCHIVE = "archive"
POOL_LATEST = "latest_slots"
SEAL_SCHEMA_VERSION = 1
NATIVE_METHODS: tuple[Method, ...] = (Method.CEN_FLAT, Method.DEGREE)
DEC_FAMILY: tuple[Method, ...] = (Method.DEC, Method.DEC_ONE_ROUND, Method.IND_PRIVATE_REVISION)


def seals_dir(run_root: str | os.PathLike, seal: str) -> Path:
    if not isinstance(seal, str) or len(seal) != 64:
        raise ValueError("seal must be the 64-hex sha256 of the sealed cells manifest")
    return Path(run_root) / SEALS_DIR / seal


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- item files


def item_path(run_root: str | os.PathLike, cell_id: str, source_id: str) -> Path:
    return Path(run_root) / "cells" / cell_id / "items" / f"{source_id}.json"


def load_item_file(run_root: str | os.PathLike, cell_id: str, source_id: str) -> dict[str, Any] | None:
    """The completed item file of ``(cell_id, source_id)`` or ``None``; a corrupt/mismatched
    file raises ``ProtocolError`` (never treated as absent)."""
    path = item_path(run_root, cell_id, source_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProtocolError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError(f"{path} is not an object")
    if data.get("source_id") != source_id or not isinstance(data.get("cell"), dict) or data["cell"].get("cell_id") != cell_id:
        raise ProtocolError(f"{path} does not describe ({cell_id}, {source_id})")
    return data


def candidate_records_of(item: Mapping[str, Any]) -> list[CandidateRecord]:
    """Every candidate opportunity of an item file (bank draws or ``episode.candidates``)."""
    if item.get("bank") is not None:
        return [CandidateRecord.from_dict(c) for c in item["bank"]]
    episode = item.get("episode")
    if not isinstance(episode, dict):
        raise ProtocolError(f"item {item.get('source_id')} has neither bank nor episode")
    return [CandidateRecord.from_dict(c) for c in episode.get("candidates", [])]


def _pool_id(cell_id: str, source_id: str, pool_kind: str, prefix_k: int | None) -> str:
    return identity.sha256_hex(identity.jcs([cell_id, source_id, pool_kind, prefix_k]))


def rekey(records: Sequence[CandidateRecord], answer_format: str) -> list[CandidateRecord]:
    """Fill ``vote_key``/``grouping_mode`` from the frozen public rule, fail-closed.

    Policies record the key of every voted candidate (WP4 ``EpisodeContext.vote``) but not
    necessarily its per-candidate grouping mode; the seal recomputes both with WP3's rule
    and raises ``ProtocolError`` when a *recorded* key or mode disagrees with it (a harness
    defect), never overwriting silently.  Invalid candidates keep ``None``.
    """
    out: list[CandidateRecord] = []
    for record in records:
        if not record.valid or record.candidate is None:
            if record.vote_key is not None:
                raise ProtocolError(f"invalid candidate {record.candidate_id} carries a vote key")
            out.append(dataclasses.replace(record, grouping_mode=None))
            continue
        key, mode = vote_key(answer_format, record.candidate.final_answer)
        if record.vote_key is not None and record.vote_key != key:
            raise ProtocolError(f"candidate {record.candidate_id} carries vote key {record.vote_key!r}; the frozen rule gives {key!r}")
        if record.grouping_mode is not None and record.grouping_mode != mode:
            raise ProtocolError(f"candidate {record.candidate_id} carries grouping mode {record.grouping_mode!r}; the frozen rule gives {mode!r}")
        out.append(dataclasses.replace(record, vote_key=key, grouping_mode=mode))
    return out


def _pool_dict(cell: CellSpec, source_id: str, item: Mapping[str, Any], pool_kind: str, records: Sequence[CandidateRecord],
               prefix_k: int | None, policy_selected: str | None, answer_format: str) -> dict[str, Any]:
    keyed = keyed_pool(rekey(records, answer_format), answer_format)
    ids = [r.candidate_id for r in keyed]
    if len(set(ids)) != len(ids):
        raise ProtocolError(f"{cell.cell_id}/{source_id}: duplicate candidate ids in pool {pool_kind}")
    return {
        "pool_id": _pool_id(cell.cell_id, source_id, pool_kind, prefix_k),
        "cell_id": cell.cell_id,
        "source_id": source_id,
        "domain": str(item.get("domain")),
        "split": str(item.get("split")),
        "answer_format": answer_format,
        "module": cell.module,
        "method": cell.method.value,
        "checkpoint": cell.checkpoint,
        "N": int(cell.N),
        "B": int(cell.B),
        "framing": cell.framing.value,
        "episode_rep": int(cell.episode_rep),
        "degree": cell.degree,
        "pool_kind": pool_kind,
        "prefix_k": prefix_k,
        "candidate_ids": ids,
        "valid": {r.candidate_id: bool(r.valid and r.candidate is not None) for r in keyed},
        "vote_keys": {r.candidate_id: r.vote_key for r in keyed},
        "grouping_modes": {r.candidate_id: r.grouping_mode for r in keyed},
        "policy_selected_candidate_id": policy_selected,
        "item_status": item.get("status"),
    }


def pools_for_item(cell: CellSpec, item: Mapping[str, Any], answer_format: str) -> list[dict[str, Any]]:
    """The public pool definitions an item file yields (see module docstring)."""
    records = candidate_records_of(item)
    sid = str(item["source_id"])
    method = Method(cell.method)
    pools: list[dict[str, Any]] = []
    if method is Method.BANK:
        if len(records) != max(PREFIXES):
            raise ProtocolError(f"{cell.cell_id}/{sid}: bank has {len(records)} draws, expected {max(PREFIXES)}")
        for k in PREFIXES:
            pools.append(_pool_dict(cell, sid, item, POOL_BANK_PREFIX, records[:k], k, None, answer_format))
        return pools
    episode = item["episode"]
    by_id = {r.candidate_id: r for r in records}
    selection = episode.get("selection")
    if method in NATIVE_METHODS:
        native = episode.get("native_final") or {}
        cid = native.get("candidate_id")
        if cid is None or cid not in by_id:
            raise ProtocolError(f"{cell.cell_id}/{sid}: native final {cid!r} is not among the candidates")
        pools.append(_pool_dict(cell, sid, item, POOL_NATIVE, [by_id[cid]], None, cid, answer_format))
        return pools
    if not isinstance(selection, dict):
        raise ProtocolError(f"{cell.cell_id}/{sid}: {method.value} item lacks its VOTE selection")
    pool_kind = str(selection["pool_kind"])
    ids = list(selection["pool_candidate_ids"])
    missing = [c for c in ids if c not in by_id]
    if missing:
        raise ProtocolError(f"{cell.cell_id}/{sid}: selection pool references unknown candidates {missing[:3]}")
    pools.append(_pool_dict(cell, sid, item, pool_kind, [by_id[c] for c in ids], None, selection.get("selected_candidate_id"), answer_format))
    if method in DEC_FAMILY and pool_kind != POOL_ARCHIVE:
        pools.append(_pool_dict(cell, sid, item, POOL_ARCHIVE, records, None, None, answer_format))
    return pools


# --------------------------------------------------------------------------- registers


def _load_register(path: Path, key: str) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SEAL_SCHEMA_VERSION, key: {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProtocolError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get(key), dict):
        raise ProtocolError(f"{path} is not a seal register")
    return data


def load_pools(run_root: str | os.PathLike, seal: str) -> dict[str, Any]:
    path = seals_dir(run_root, seal) / POOLS_FILE
    if not path.exists():
        raise ProtocolError(f"pools are not sealed: {path} is missing (run seal --kind pools first)")
    return _load_register(path, "pools")


def load_selections(run_root: str | os.PathLike, seal: str) -> dict[str, Any]:
    path = seals_dir(run_root, seal) / SELECTIONS_FILE
    if not path.exists():
        raise ProtocolError(f"selections are not sealed: {path} is missing (run seal --kind selections first)")
    return _load_register(path, "selections")


def sealed_item_ids(pools: Mapping[str, Any]) -> set[str]:
    return {p["source_id"] for p in pools["pools"].values()}


def pools_of_item(pools: Mapping[str, Any], source_id: str) -> list[dict[str, Any]]:
    return sorted((p for p in pools["pools"].values() if p["source_id"] == source_id), key=lambda p: p["pool_id"])


def selections_of_item(selections: Mapping[str, Any], source_id: str) -> list[dict[str, Any]]:
    return sorted((s for s in selections["selections"].values() if s["source_id"] == source_id), key=lambda s: s["selection_id"])


def assert_item_sealed(selections: Mapping[str, Any], source_id: str) -> list[dict[str, Any]]:
    """T18: the sealed selections of an item, or ``ProtocolError`` when none exist."""
    found = selections_of_item(selections, source_id)
    if not found:
        raise ProtocolError(f"{source_id}: no sealed selection; evaluation refuses to run (§3.6, P0-3)")
    return found


def sealed_candidate_ids(selections: Mapping[str, Any], source_id: str) -> set[str]:
    """Union of the sealed pools' candidate ids of an item (the evaluable set)."""
    out: set[str] = set()
    for sel in assert_item_sealed(selections, source_id):
        out.update(sel["record"]["pool_candidate_ids"])
    return out


def load_pool_candidates(run_root: str | os.PathLike, pools: Sequence[Mapping[str, Any]]) -> dict[str, CandidateRecord]:
    """``{candidate_id: CandidateRecord}`` for every candidate named by ``pools`` (read from the
    producing cells' item files; a missing file or id is a ``ProtocolError``)."""
    out: dict[str, CandidateRecord] = {}
    cache: dict[tuple[str, str], dict[str, CandidateRecord]] = {}
    for pool in pools:
        key = (str(pool["cell_id"]), str(pool["source_id"]))
        if key not in cache:
            item = load_item_file(run_root, *key)
            if item is None:
                raise ProtocolError(f"sealed pool {pool['pool_id'][:12]} references a missing item file {key}")
            cache[key] = {r.candidate_id: r for r in candidate_records_of(item)}
        for cid in pool["candidate_ids"]:
            rec = cache[key].get(cid)
            if rec is None:
                raise ProtocolError(f"sealed pool {pool['pool_id'][:12]}: candidate {cid[:12]} absent from {key}")
            out[cid] = rec
    return out


def _merge(register: dict[str, Any], key: str, entries: Mapping[str, dict[str, Any]], now: float, what: str) -> tuple[int, int]:
    """Append new entries (stamped ``sealed_at``); re-assert existing ones byte-for-byte."""
    added = kept = 0
    for eid, entry in entries.items():
        existing = register[key].get(eid)
        if existing is None:
            register[key][eid] = {**entry, "sealed_at": now, "sealed_at_iso": _iso(now)}
            added += 1
            continue
        stripped = {k: v for k, v in existing.items() if k not in ("sealed_at", "sealed_at_iso")}
        if stripped != entry:
            raise ProtocolError(f"{what} {eid[:12]} already sealed with different content; seals are immutable")
        kept += 1
    return added, kept


def _write_register(path: Path, register: Mapping[str, Any]) -> None:
    io.atomic_write_text(path, json.dumps(register, indent=1, sort_keys=True) + "\n")


def _generate_cells(cells: Sequence[CellSpec]) -> list[CellSpec]:
    out = [c for c in cells if c.kind is CellKind.GENERATE]
    if not out:
        raise ProtocolError("the cells file holds no generate cells to seal")
    return out


def seal_pools(run_root: str | os.PathLike, cells_file: str | os.PathLike, *, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Seal every completed item's pools of the generate cells in ``cells_file`` (append-only)."""
    run_root = Path(run_root)
    cells_file = Path(cells_file)
    seal = cells_file_sha256(cells_file)
    cells = _generate_cells(load_cells_file(cells_file))
    from agents_scaling.study.data.public import load_public_tasks  # public view only

    formats = {t.source_id: t.answer_format for t in load_public_tasks(run_root)}
    path = seals_dir(run_root, seal) / POOLS_FILE
    register = _load_register(path, "pools")
    register.setdefault("seal", seal)
    register.setdefault("cells_file", str(cells_file))
    if register["seal"] != seal:
        raise ProtocolError(f"{path} belongs to seal {register['seal'][:12]}, not {seal[:12]}")
    entries: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []
    for cell in cells:
        for sid in cell.items:
            item = load_item_file(run_root, cell.cell_id, sid)
            if item is None:
                incomplete.append(f"{cell.cell_id}/{sid}")
                continue
            if sid not in formats:
                raise ProtocolError(f"{sid} is not in the public export")
            for pool in pools_for_item(cell, item, formats[sid]):
                entries[pool["pool_id"]] = pool
    now = clock()
    added, kept = _merge(register, "pools", entries, now, "pool")
    register["incomplete"] = sorted(incomplete)
    register["last_sealed_at"] = now
    register["n_pools"] = len(register["pools"])
    register["n_items"] = len(sealed_item_ids(register))
    _write_register(path, register)
    return {"seal": seal, "path": str(path), "added": added, "kept": kept, "incomplete": len(incomplete), "n_items": register["n_items"]}


def _judge_best_scores(run_root: Path, select_cells: Sequence[CellSpec], seal: str) -> dict[str, dict[str, float | None]]:
    """``{source_id: {candidate_id: score|None}}`` from the JUDGE_BEST cells of ``seal``."""
    scores: dict[str, dict[str, float | None]] = {}
    for cell in select_cells:
        if cell.kind is not CellKind.JUDGE_BEST:
            continue
        if cell_seal(cell) != seal:
            raise ProtocolError(f"select cell {cell.cell_id} scores seal {cell_seal(cell)}, not {seal[:12]}")
        for sid in cell.items:
            item = load_item_file(run_root, cell.cell_id, sid)
            if item is None:
                continue
            if item.get("kind") != CellKind.JUDGE_BEST.value or not isinstance(item.get("scores"), dict):
                raise ProtocolError(f"{cell.cell_id}/{sid}: not a JUDGE_BEST item file")
            bucket = scores.setdefault(sid, {})
            for cid, score in item["scores"].items():
                if cid in bucket and bucket[cid] != score:
                    raise ProtocolError(f"{sid}: conflicting JUDGE_BEST scores for {cid[:12]}")
                bucket[cid] = None if score is None else float(score)
    return scores


def _selection_entry(pool: Mapping[str, Any], record: SelectionRecord, score_cell: str | None) -> dict[str, Any]:
    return {
        "selection_id": identity.sha256_hex(identity.jcs([pool["pool_id"], record.selector_id])),
        "pool_id": pool["pool_id"],
        "selector_id": record.selector_id,
        "cell_id": pool["cell_id"],
        "source_id": pool["source_id"],
        "domain": pool["domain"],
        "module": pool["module"],
        "method": pool["method"],
        "checkpoint": pool["checkpoint"],
        "N": pool["N"],
        "B": pool["B"],
        "framing": pool["framing"],
        "episode_rep": pool["episode_rep"],
        "degree": pool["degree"],
        "pool_kind": pool["pool_kind"],
        "prefix_k": pool["prefix_k"],
        "record": record.to_dict(),
        "score_cell_id": score_cell,
    }


def seal_selections(
    run_root: str | os.PathLike,
    cells_file: str | os.PathLike,
    *,
    cfg: StudyConfig | None = None,
    select_cells_file: str | os.PathLike | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Seal VOTE (and JUDGE_BEST where scores exist) selections over the sealed pools.

    VOTE is recomputed with WP3's frozen rule over the sealed candidates; for the policy's
    own primary pool the recomputed winner must equal the recorded one.  JUDGE_BEST needs
    the scores of every valid candidate of a pool (from ``select_cells_file`` item files);
    a pool without complete scores gets no JUDGE_BEST selection (reported).
    """
    run_root = Path(run_root)
    cfg = cfg or load_config()
    seal = cells_file_sha256(Path(cells_file))
    pools = load_pools(run_root, seal)
    scores = _judge_best_scores(run_root, load_cells_file(select_cells_file), seal) if select_cells_file else {}
    score_cell: dict[str, str] = {}
    if select_cells_file:
        for cell in load_cells_file(select_cells_file):
            for sid in cell.items:
                score_cell[sid] = cell.cell_id
    path = seals_dir(run_root, seal) / SELECTIONS_FILE
    register = _load_register(path, "selections")
    register.setdefault("seal", seal)
    if register["seal"] != seal:
        raise ProtocolError(f"{path} belongs to seal {register['seal'][:12]}, not {seal[:12]}")
    entries: dict[str, dict[str, Any]] = {}
    unscored = 0
    all_pools = sorted(pools["pools"].values(), key=lambda p: p["pool_id"])
    candidates = load_pool_candidates(run_root, all_pools)
    for pool in all_pools:
        sid, fmt = pool["source_id"], pool["answer_format"]
        records = rekey([candidates[cid] for cid in pool["candidate_ids"]], fmt)
        if pool["pool_kind"] == POOL_BANK_PREFIX:
            bank_records = records
            rec = prefix_vote(bank_records, int(pool["prefix_k"]), sid, cfg.study_seed, fmt)
        else:
            rec = vote(records, sid, cfg.study_seed, fmt, pool_kind=pool["pool_kind"])
        expected = pool.get("policy_selected_candidate_id")
        if pool["pool_kind"] not in (POOL_BANK_PREFIX, POOL_NATIVE) and expected is not None and rec.selected_candidate_id != expected:
            raise ProtocolError(f"{pool['cell_id']}/{sid}: recomputed VOTE {rec.selected_candidate_id} != recorded {expected}")
        entry = _selection_entry(pool, rec, None)
        entries[entry["selection_id"]] = entry
        item_scores = scores.get(sid)
        valid_ids = [cid for cid in pool["candidate_ids"] if pool["valid"][cid]]
        if item_scores is not None and all(cid in item_scores for cid in valid_ids):
            jb = judge_best_select({cid: item_scores[cid] for cid in valid_ids}, records, sid, cfg.study_seed, fmt,
                                   pool_kind=pool["pool_kind"], prefix_k=pool["prefix_k"])
            entry = _selection_entry(pool, jb, score_cell.get(sid))
            entries[entry["selection_id"]] = entry
        elif select_cells_file:
            unscored += 1
    now = clock()
    added, kept = _merge(register, "selections", entries, now, "selection")
    register["last_sealed_at"] = now
    register["n_selections"] = len(register["selections"])
    register["n_items"] = len({s["source_id"] for s in register["selections"].values()})
    register["pools_without_judge_best"] = unscored
    _write_register(path, register)
    return {"seal": seal, "path": str(path), "added": added, "kept": kept, "unscored_pools": unscored, "n_items": register["n_items"]}


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agents_scaling.study.seal", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--cells-file", required=True, help="the generate cells manifest to seal (absolute or under the run root)")
    p.add_argument("--kind", required=True, choices=("pools", "selections"))
    p.add_argument("--select-cells-file", default=None, help="selections: the JUDGE_BEST cells manifest whose scores enter")
    p.add_argument("--config", default=None)
    p.add_argument("--results-root", default=os.environ.get("ASYS_RESULTS_ROOT", DEFAULT_RESULTS_ROOT))
    return p


def _resolve(run_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else run_root / path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.results_root) / args.run_id
    cells_file = _resolve(run_root, args.cells_file)
    if args.kind == "pools":
        out = seal_pools(run_root, cells_file)
    else:
        out = seal_selections(run_root, cells_file, cfg=load_config(args.config), select_cells_file=_resolve(run_root, args.select_cells_file))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [name for name in globals() if not name.startswith("_")]
