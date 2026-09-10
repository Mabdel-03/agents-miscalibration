"""ARD36 neural capture: a decision-manifest-driven replay runner (WP6).

Spec §14.2 (capture cohort), §14.5 (replay engineering); plan §5.6.

What is reused, unchanged
-------------------------
``study.neural.engine.CaptureEngine`` (hooks block **outputs**, i.e. the residual stream
after block ``b`` = the tensor entering ``b+1``; ``blocks_for(36) == (8, 17, 26)`` is
already the spec's ``floor(L/4)-1`` triple for Qwen3-8B; ``mask_policy='causal'`` is kept
because a padded 2-D mask crashed SDPA on the 32B), ``study.neural.replay``
(``teacher_forced_sequence``, ``check_prompt_identity``, ``fidelity``) and
``study.neural.storage`` (``ShardWriter``: atomic chunked fp16 storage, resume by
committed key; a missing anchor is a ledger row with **no vector**, never a zero vector).

What is replaced
----------------
``study.neural.capture``'s work-list machinery is study_v4-specific: its HMAC group
selection (``select_native_calls``), ``GEOMETRY_GROUPS`` and cells/episode manifests all
speak the v4 method vocabulary.  ARD36 drives the same inner loop from a **decision
manifest** instead — one row per decision to replay — and keeps ``run_native``'s shape:
resume filter → build ``CaptureRequest`` → forward → write.

ARD36 specifics
---------------
* Join keys ``decision_id, episode_id, agent_id, round, protocol, cohort,
  boundary_injected`` travel in ``ActivationRow.extra`` (plan §5.6), so a captured vector
  can be joined to ``decisions.parquet`` and to the fork manifest without a second index.
* **Full-vocabulary next-token entropy is computed in the same pass** at the two
  ``contract.ENTROPY_ANCHORS`` only, through ``CaptureRequest.logit_positions`` — the
  engine applies ``lm_head`` at those positions only and never materialises
  ``[B, T, vocab]``.  Only the **scalar** entropy is persisted (spec §14.5: "Do not persist
  full vocabulary logits"; "top-k logprobs are not full entropy").
* The frozen work list is ordered **train/val first**, then test (plan §5.6: the 80
  train/val items are dispatched at fleet-ready + ~2 h, the 64 test items follow as their
  episodes complete).
* Shards are staged on node-local ``/tmp`` and consolidated atomically into the run root
  (spec §14.5: "Put scratch shards on local storage and consolidate atomically").  The
  ``.npz`` is moved before its ``.jsonl``, because ``storage.list_chunks`` treats the
  ``.jsonl`` as the commit marker and refuses a chunk whose ``.npz`` is missing.
* The parity panel (spec §14.5, plan §5.6) is stratified across root and rounds 1-3 and
  all three captured protocols and reported per round, and the byte-identical-root sanity
  check (spec §14.2) compares the five round-0 ``PROMPT_END`` vectors of one item.

Owned by WP6 (``docs/ard36/00_contract.md`` §1).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from agents_scaling.ard import contract as K
from agents_scaling.ard.neural import anchors as AA
from agents_scaling.study import identity
from agents_scaling.study.neural import storage as S
from agents_scaling.study.neural.anchors import AnchorError
from agents_scaling.study.neural.engine import HOOK_CONVENTION, CaptureRequest
from agents_scaling.study.neural.replay import (
    check_prompt_identity,
    fidelity,
    summarize_fidelity,
    teacher_forced_sequence,
)

__all__ = [
    "CAPTURE_STAGE",
    "CaptureWorkItem",
    "JOIN_KEYS",
    "SPLIT_ORDER",
    "consolidate_shards",
    "load_manifest",
    "next_token_entropy",
    "parity_panel",
    "parity_panel_items",
    "pending_items",
    "root_prefill_identity",
    "manifest_for_item",
    "manifest_from_item_file",
    "run_capture",
    "run_capture_cell",
    "shard_of",
    "staging_root",
    "work_list",
]

#: ``storage.stage_dir`` only accepts the two frozen stage names; a single-request native
#: budget capture is exactly the "native" stage (ARD36 has its own run root, so there is
#: no collision with study_v4's rows).
CAPTURE_STAGE = "native"

#: Plan §5.6: the frozen work list is train/val first, then test.
SPLIT_ORDER: Mapping[str, int] = {"train": 0, "validation": 1, "val": 1, "test": 2}

#: The join keys every stored row carries in ``ActivationRow.extra`` (plan §5.6).
JOIN_KEYS: tuple[str, ...] = (
    "decision_id", "episode_id", "agent_id", "round", "protocol", "cohort", "boundary_injected",
)


def log(message: str) -> None:
    print(f"[ard36.neural] {message}", flush=True)


# --------------------------------------------------------------------------- work list


@dataclass(frozen=True)
class CaptureWorkItem:
    """One decision to replay.  ``decision_id == request_id`` (``ard.episode.Decision``)."""

    decision_id: str
    episode_id: str
    item_id: str
    agent_id: int
    round_: int
    protocol: str
    cohort: str = K.Cohort.CAPTURE.value
    split: str = ""
    family: str = ""
    checkpoint: str = K.CHECKPOINT_SMALL
    boundary_injected: bool = False
    cell_id: str = ""
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("a capture work item needs a decision_id")
        if not 0 <= int(self.agent_id) < K.N_AGENTS:
            raise ValueError(f"agent_id {self.agent_id} outside 0..{K.N_AGENTS - 1}")
        if int(self.round_) not in K.ROUNDS:
            raise ValueError(f"round {self.round_} outside {K.ROUNDS}")
        if not self.request_id:
            object.__setattr__(self, "request_id", self.decision_id)

    @property
    def order_key(self) -> tuple[int, str, str, int, int]:
        """Train/val before test, then a deterministic order inside each split."""
        return (SPLIT_ORDER.get(str(self.split), len(SPLIT_ORDER)), str(self.item_id),
                str(self.protocol), int(self.round_), int(self.agent_id))

    @property
    def join(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "episode_id": self.episode_id,
            "agent_id": int(self.agent_id),
            "round": int(self.round_),
            "protocol": str(self.protocol),
            "cohort": str(self.cohort),
            "boundary_injected": bool(self.boundary_injected),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CaptureWorkItem":
        """A manifest row (``decisions.parquet`` field names) → a work item."""
        def pick(*names: str, default: Any = "") -> Any:
            for name in names:
                if name in row and row[name] is not None:
                    return row[name]
            return default

        return cls(
            decision_id=str(pick("decision_id", "request_id", "request_hash")),
            episode_id=str(pick("episode_id")),
            item_id=str(pick("item_id", "source_id")),
            agent_id=int(pick("agent_id", "agent_slot", default=0)),
            round_=int(pick("round", "round_", default=0)),
            protocol=str(pick("protocol")),
            cohort=str(pick("cohort", default=K.Cohort.CAPTURE.value)),
            split=str(pick("split")),
            family=str(pick("family")),
            checkpoint=str(pick("checkpoint", default=K.CHECKPOINT_SMALL)),
            boundary_injected=bool(pick("boundary_injected", default=False)),
            cell_id=str(pick("cell_id")),
            request_id=str(pick("request_id", "request_hash", "decision_id")),
        )


def work_list(rows: Iterable[Mapping[str, Any] | CaptureWorkItem]) -> tuple[CaptureWorkItem, ...]:
    """The frozen, deduplicated work list: train/val first, then test (plan §5.6).

    Duplicate ``decision_id`` rows are collapsed (root captures are reused across
    protocols, spec §14.2), keeping the first occurrence in sorted order.
    """
    items = [r if isinstance(r, CaptureWorkItem) else CaptureWorkItem.from_row(r) for r in rows]
    items.sort(key=lambda w: w.order_key)
    seen: set[str] = set()
    out: list[CaptureWorkItem] = []
    for item in items:
        if item.decision_id in seen:
            continue
        seen.add(item.decision_id)
        out.append(item)
    return tuple(out)


def load_manifest(path: str | os.PathLike) -> tuple[CaptureWorkItem, ...]:
    """Read a decision manifest: ``.jsonl`` (one row per line), ``.json`` (a list or
    ``{"decisions": [...]}``) or ``.parquet``."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(p).to_dict(orient="records")
    elif p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload["decisions"] if isinstance(payload, Mapping) else payload
    return work_list(rows)


def shard_of(items: Sequence[CaptureWorkItem], shard: int, num_shards: int) -> list[CaptureWorkItem]:
    """Round-robin shard of a frozen work list (order-preserving, disjoint, complete)."""
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard}/{num_shards} is not a valid split")
    return [w for i, w in enumerate(items) if i % num_shards == shard]


def pending_items(
    items: Sequence[CaptureWorkItem],
    done: set[tuple[str, int, str]],
    blocks: Sequence[int],
    kinds: Sequence[str] = K.CAPTURE_ANCHORS,
) -> list[CaptureWorkItem]:
    """Resume filter: drop work items whose every (block, anchor) key is already committed."""
    out: list[CaptureWorkItem] = []
    for w in items:
        if all((w.decision_id, int(b), k) in done for b in blocks for k in kinds):
            continue
        out.append(w)
    return out


# --------------------------------------------------------------------------- entropy


def next_token_entropy(logits: np.ndarray) -> dict[str, float]:
    """Full-vocabulary next-token entropy of one logit row, as scalars (spec §14.5).

    Computed in float64 through the log-sum-exp shift; the vocabulary distribution itself
    is discarded here and is never written to disk.  ``top1_prob`` and ``logit_margin`` are
    scalars of the same distribution and are kept as diagnostics — they are explicitly
    *not* the entropy and are named accordingly.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if z.size < 2 or not np.isfinite(z).all():
        raise ValueError("next_token_entropy needs a finite logit row over a real vocabulary")
    shifted = z - z.max()
    exp = np.exp(shifted)
    total = exp.sum()
    p = exp / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    order = np.partition(z, -2)[-2:]
    return {
        "entropy_nats": entropy,
        "entropy_bits": entropy / float(np.log(2.0)),
        "vocab_size": int(z.size),
        "top1_prob": float(p.max()),
        "logit_margin": float(order[1] - order[0]),
    }


# --------------------------------------------------------------------------- staging


def staging_root(base: str | os.PathLike | None = None, *, tag: str = "ard36") -> Path:
    """A node-local staging directory for shard chunks (spec §14.5).

    ``base`` defaults to ``$ARD36_LOCAL_SCRATCH``, then ``$TMPDIR``, then ``/tmp`` — the
    node-local disk of the replay job, never the shared filesystem.
    """
    if base is None:
        base = os.environ.get("ARD36_LOCAL_SCRATCH") or os.environ.get("TMPDIR") or "/tmp"
    root = Path(tempfile.mkdtemp(prefix=f"{tag}.neural.", dir=str(base)))
    return root


def _move_atomic(src: Path, dst: Path) -> None:
    """``src`` → ``dst`` atomically, across filesystems if need be (copy + ``os.replace``)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
        return
    except OSError:
        pass  # EXDEV: node-local /tmp and the run root are different filesystems
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, dst)
        src.unlink(missing_ok=True)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def consolidate_shards(staging: str | os.PathLike, run_root: str | os.PathLike, *, stage: str = CAPTURE_STAGE) -> list[Path]:
    """Move every complete chunk of a staging run root into the real one, atomically.

    Chunks are renumbered from the destination's first free index, so two staged shards of
    the same name never collide.  The ``.npz`` lands before its ``.jsonl`` (the commit
    marker), so a crash mid-consolidation leaves an orphan ``.npz`` — which
    ``storage.next_chunk_index`` already skips — and never a chunk without vectors.
    """
    source = S.stage_dir(staging, stage)
    destination = S.stage_dir(run_root, stage)
    destination.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    next_index: dict[str, int] = {}
    for shard, _index, npz, jsonl in S.list_chunks(source):
        if shard not in next_index:
            next_index[shard] = S.next_chunk_index(destination, shard)
        target_npz, target_jsonl = S.chunk_paths(destination, shard, next_index[shard])
        next_index[shard] += 1
        _move_atomic(npz, target_npz)
        _move_atomic(jsonl, target_jsonl)
        moved.append(target_jsonl)
    for extra in sorted(source.glob("*.json")):
        _move_atomic(extra, destination / extra.name)
    return moved


# --------------------------------------------------------------------------- the runner


@dataclass
class CaptureConfig:
    """Everything ``run_capture`` needs that is not the engine, tokenizer or store."""

    revision: str
    checkpoint: str = K.CHECKPOINT_SMALL
    shard_name: str = "ard36.s000"
    chunk_rows: int = 128
    group_size: int = 8
    anchors: tuple[str, ...] = K.CAPTURE_ANCHORS
    entropy_anchors: frozenset[str] = field(default_factory=lambda: frozenset(K.ENTROPY_ANCHORS))
    strict_records: bool = True
    check_identity: bool = True
    allow_model_mismatch: bool = False
    fidelity_records: int = 0
    fidelity_tokens: int = 20


def run_capture(
    items: Sequence[CaptureWorkItem],
    *,
    engine: Any,
    tokenizer: Any,
    store: Any,
    run_root: str | os.PathLike,
    config: CaptureConfig,
    staging: str | os.PathLike | None = None,
    on_event: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Replay every pending decision and write its four anchors × every block.

    Same inner shape as ``study.neural.capture.run_native``: resume filter, build one
    ``CaptureRequest`` per decision (positions = the present anchors; ``logit_positions`` =
    the present *entropy* anchors), forward the group, write rows.  Returns the stats dict
    and writes it next to the shards.

    ``staging`` (default: a fresh node-local directory) is where chunks are written; they
    are consolidated into ``run_root`` before this function returns.
    """
    run_root = Path(run_root)
    blocks = [int(b) for b in engine.blocks]
    kinds = tuple(config.anchors)
    done = S.existing_keys(run_root, CAPTURE_STAGE)
    pending = pending_items(items, done, blocks, kinds)
    owned_staging = staging is None
    staging_path = Path(staging) if staging is not None else staging_root()
    stats: dict[str, Any] = {
        "work_items": len(items), "pending": len(pending), "captured": 0, "rows": 0,
        "tokens": 0, "missing_records": 0, "anchor_errors": 0, "rows_skipped_committed": 0,
        "anchors_missing": {}, "entropy_rows": 0, "blocks": blocks,
        "anchors": list(kinds), "entropy_anchors": sorted(config.entropy_anchors),
        "staging": str(staging_path), "shard_name": config.shard_name,
    }
    fidelity_reports: list[dict[str, Any]] = []
    log(f"{len(items)} work items, {len(pending)} pending ({len(done)} keys already stored)")

    try:
        with S.ShardWriter(staging_path, CAPTURE_STAGE, config.shard_name, chunk_rows=config.chunk_rows) as writer:
            for group in _chunks(pending, config.group_size):
                requests: list[CaptureRequest] = []
                meta: list[tuple[CaptureWorkItem, Any, AA.AnchorSet, str]] = []
                for w in group:
                    record = store.get(w.request_id)
                    if record is None:
                        stats["missing_records"] += 1
                        if config.strict_records:
                            raise RuntimeError(f"request record {w.request_id} not in the store")
                        log(f"missing record {w.request_id[:12]} (skipped)")
                        continue
                    model_rev = _model_revision(record)
                    if config.revision and model_rev != config.revision and not config.allow_model_mismatch:
                        raise RuntimeError(
                            f"{w.request_id[:12]}: record model_revision {model_rev[:8]} != engine "
                            f"{config.revision[:8]} (set allow_model_mismatch to override)"
                        )
                    if config.check_identity:
                        check_prompt_identity(record, tokenizer, strict=True)
                    try:
                        anchor_set = AA.resolve_decision_anchors(record, tokenizer, decision_id=w.decision_id)
                    except AnchorError as exc:
                        stats["anchor_errors"] += 1
                        log(f"anchor error on {w.decision_id[:12]}: {exc} (recorded as missing)")
                        _write_unresolved(writer, w, exc, blocks=blocks, kinds=kinds, done=done,
                                          config=config, engine=engine, stats=stats, model_rev=model_rev,
                                          record=record)
                        continue
                    seq = teacher_forced_sequence(record, anchor_set.k_max)
                    positions = anchor_set.present_offsets
                    logit_positions = AA.entropy_anchor_offsets(
                        [a for a in anchor_set.anchors if a.kind in config.entropy_anchors]
                    )
                    requests.append(CaptureRequest(w.decision_id, seq, positions, logit_positions))
                    meta.append((w, record, anchor_set, model_rev))
                    if config.fidelity_records and len(fidelity_reports) < config.fidelity_records:
                        fidelity_reports.append(fidelity(record, engine, config.fidelity_tokens, tokenizer=tokenizer))
                if not requests:
                    continue
                results = engine.forward(requests)
                by_id = {r.seq_id: r for r in results}
                for w, record, anchor_set, model_rev in meta:
                    result = by_id[w.decision_id]
                    _write_decision(writer, w, record, anchor_set, result, blocks=blocks, done=done,
                                    config=config, engine=engine, stats=stats, model_rev=model_rev)
                    stats["captured"] += 1
                    stats["tokens"] += int(result.n_tokens)
                writer.flush()
                if on_event is not None:
                    on_event("group_done", {"captured": stats["captured"], "rows": stats["rows"]})
                log(f"captured {stats['captured']}/{len(pending)} decisions, {stats['tokens']} tokens")
        if fidelity_reports:
            summary = summarize_fidelity(fidelity_reports)
            S.atomic_write_json(S.stage_dir(staging_path, CAPTURE_STAGE) / f"fidelity.{config.shard_name}.json",
                                {"summary": summary, "tokens_per_record": config.fidelity_tokens,
                                 "records": fidelity_reports})
            stats["fidelity"] = summary
        S.atomic_write_json(S.stage_dir(staging_path, CAPTURE_STAGE) / f"stats.{config.shard_name}.json", stats)
        stats["consolidated"] = [str(p) for p in consolidate_shards(staging_path, run_root, stage=CAPTURE_STAGE)]
    finally:
        if owned_staging:
            shutil.rmtree(staging_path, ignore_errors=True)
    return stats


def _chunks(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def _model_revision(record: Any) -> str:
    if isinstance(record, Mapping):
        block = record.get("identity") or {}
        return str(block.get("model_revision") or (record.get("model") or {}).get("model_revision") or "")
    return str((record.identity or {}).get("model_revision") or (record.model or {}).get("model_revision") or "")


def _nonce(record: Any) -> str:
    if isinstance(record, Mapping):
        block = record.get("identity") or {}
        nonce = block.get("input_hash")
        if nonce:
            return str(nonce)
        return identity.input_hash(record["messages"], record.get("chat_template_kwargs") or {})
    nonce = (record.identity or {}).get("input_hash")
    if nonce:
        return str(nonce)
    return identity.input_hash(record.messages, record.chat_template_kwargs or {})


def _extra(w: CaptureWorkItem, record: Any, anchor_set: AA.AnchorSet | None, model_rev: str) -> dict[str, Any]:
    extra = dict(w.join)
    extra.update({
        "item_id": w.item_id,
        "split": w.split,
        "family": w.family,
        "cell_id": w.cell_id,
        "record_model_revision": model_rev,
        "prompt_tokens": None if anchor_set is None else int(anchor_set.prompt_tokens),
        "completion_tokens": None if anchor_set is None else int(anchor_set.completion_tokens),
    })
    if anchor_set is not None:
        extra["markers"] = dict(anchor_set.markers)
        extra["report_end_source"] = anchor_set.report_end_source
        extra["truncated"] = bool(anchor_set.truncated)
        extra["k_max"] = int(anchor_set.k_max)
    return extra


def _row(
    *, w: CaptureWorkItem, anchor: AA.TagAnchor, block: int, result: Any, nonce: str,
    revision: str, checkpoint: str, hidden_size: int, extra: Mapping[str, Any],
) -> tuple[S.ActivationRow, np.ndarray | None]:
    vector = result.residual_at(block, anchor.token_offset) if (anchor.present and result is not None) else None
    row = S.ActivationRow(
        StateSnapshot_id=w.decision_id,
        consumer_revision=revision,
        condition=f"ard:{w.protocol}:r{int(w.round_)}",
        child_slot=None,
        nonce_hash=nonce,
        block=int(block),
        anchor_kind=anchor.kind,
        structural_span=anchor.structural_span,
        token_offset=anchor.token_offset,
        channel=anchor.channel,
        generated_token_count=int(anchor.generated_token_count),
        missingness=anchor.missingness,
        tensor_hash=None,
        measurement_cost=(result.measurement_cost if result is not None
                          else {"sequence_tokens": 0, "share_seconds": 0.0}),
        operationally_available_at_checkpoint=True,
        stage=CAPTURE_STAGE,
        sequence_id=w.decision_id,
        source_id=w.item_id,
        method=str(w.protocol),
        role=f"agent{int(w.agent_id)}",
        phase=f"round{int(w.round_)}",
        cell_id=w.cell_id or None,
        episode_id=w.episode_id or None,
        checkpoint=checkpoint,
        hidden_size=int(hidden_size),
        sequence_tokens=None if result is None else int(result.n_tokens),
        hook_convention=HOOK_CONVENTION,
        extra={
            **dict(extra),
            "anchor_detail": anchor.detail,
            "byte_offset": anchor.byte_offset,
            "feature_time": AA.feature_time_of(anchor.kind),
            "last_allowed_round": AA.last_allowed_round(anchor.kind, w.round_),
        },
        captured_at=time.time(),
    )
    return row, vector


def _write_decision(
    writer: S.ShardWriter, w: CaptureWorkItem, record: Any, anchor_set: AA.AnchorSet, result: Any,
    *, blocks: Sequence[int], done: set[tuple[str, int, str]], config: CaptureConfig,
    engine: Any, stats: dict[str, Any], model_rev: str,
) -> None:
    nonce = _nonce(record)
    extra = _extra(w, record, anchor_set, model_rev)
    entropies: dict[str, dict[str, float]] = {}
    for anchor in anchor_set.anchors:
        if anchor.present and anchor.kind in config.entropy_anchors:
            # Same pass, designated anchors only; the vocabulary row itself never leaves here.
            entropies[anchor.kind] = next_token_entropy(result.logits_at(anchor.token_offset))
    for anchor in anchor_set.anchors:
        if anchor.kind not in config.anchors:
            continue
        if not anchor.present:
            stats["anchors_missing"][anchor.missingness] = stats["anchors_missing"].get(anchor.missingness, 0) + 1
        row_extra = dict(extra)
        if anchor.kind in entropies:
            row_extra.update(entropies[anchor.kind])
            row_extra["entropy_anchor"] = True
        else:
            row_extra["entropy_anchor"] = False
        for b in blocks:
            if (w.decision_id, int(b), anchor.kind) in done:
                stats["rows_skipped_committed"] += 1
                continue
            row, vector = _row(w=w, anchor=anchor, block=b, result=result, nonce=nonce,
                               revision=config.revision, checkpoint=config.checkpoint,
                               hidden_size=engine.hidden_size, extra=row_extra)
            writer.add(row, vector)
            stats["rows"] += 1
            if anchor.kind in entropies:
                stats["entropy_rows"] += 1


def _write_unresolved(
    writer: S.ShardWriter, w: CaptureWorkItem, exc: Exception, *, blocks: Sequence[int],
    kinds: Sequence[str], done: set[tuple[str, int, str]], config: CaptureConfig, engine: Any,
    stats: dict[str, Any], model_rev: str, record: Any,
) -> None:
    """A decision whose anchors cannot be derived is stored as typed missing rows, never
    dropped (study_v4 R2 rule, kept: the ledger must account for every work item)."""
    nonce = _nonce(record)
    extra = {**_extra(w, record, None, model_rev), "anchor_error": str(exc)[:300]}
    for kind in kinds:
        anchor = AA.TagAnchor(kind=kind, token_offset=None, missingness=AA.MISSING_ANCHOR_UNRESOLVED,
                              channel=AA.CHANNEL_NONE, structural_span="", generated_token_count=0,
                              detail={"error": str(exc)[:300]})
        stats["anchors_missing"][AA.MISSING_ANCHOR_UNRESOLVED] = (
            stats["anchors_missing"].get(AA.MISSING_ANCHOR_UNRESOLVED, 0) + 1)
        for b in blocks:
            if (w.decision_id, int(b), kind) in done:
                stats["rows_skipped_committed"] += 1
                continue
            row, vector = _row(w=w, anchor=anchor, block=b, result=None, nonce=nonce,
                               revision=config.revision, checkpoint=config.checkpoint,
                               hidden_size=engine.hidden_size, extra=extra)
            writer.add(row, vector)
            stats["rows"] += 1


# --------------------------------------------------------------------------- parity panel


def parity_panel_items(
    items: Sequence[CaptureWorkItem],
    *,
    per_cell: int = 2,
    protocols: Sequence[str] = tuple(p.value for p in K.CAPTURE_PROTOCOLS),
    rounds: Sequence[int] = K.ROUNDS,
) -> list[CaptureWorkItem]:
    """A panel stratified across root and rounds 1-3 × the three captured protocols.

    Round 0 is the common neutral root of every protocol (contract: "Round 0 is the common
    neutral root team for all of them"), so the round-0 cell is filled from whatever
    protocol the root decision was recorded under.  Selection is the frozen work-list
    order, so the panel is reproducible and correctness-blind.
    """
    chosen: list[CaptureWorkItem] = []
    for round_ in rounds:
        if int(round_) == 0:
            cell = [w for w in items if int(w.round_) == 0]
            chosen.extend(cell[: per_cell])
            continue
        for protocol in protocols:
            cell = [w for w in items if int(w.round_) == int(round_) and str(w.protocol) == str(protocol)]
            chosen.extend(cell[: per_cell])
    return chosen


def parity_panel(
    items: Sequence[CaptureWorkItem],
    *,
    engine: Any,
    tokenizer: Any,
    store: Any,
    per_cell: int = 2,
    n_tokens: int = 20,
) -> dict[str, Any]:
    """Spec §14.5 replay parity, reported **per round** (plan §5.6).

    For every panel decision: argmax agreement with the stored sampled tokens, mean
    log-prob of the stored tokens, and the two-pass logit discrepancy (all from
    ``study.neural.replay.fidelity``, reused).  A sampled completion need not agree with
    the argmax, so agreement is descriptive; the two-pass discrepancy is the parity check.
    """
    panel = parity_panel_items(items, per_cell=per_cell)
    reports: list[dict[str, Any]] = []
    for w in panel:
        record = store.get(w.request_id)
        if record is None:
            continue
        report = fidelity(record, engine, n_tokens, tokenizer=tokenizer)
        report.update({"round": int(w.round_), "protocol": str(w.protocol),
                       "decision_id": w.decision_id, "item_id": w.item_id})
        reports.append(report)
    per_round: dict[str, Any] = {}
    for round_ in sorted({int(r["round"]) for r in reports}):
        subset = [r for r in reports if int(r["round"]) == round_]
        per_round[str(round_)] = {**summarize_fidelity(subset), "protocols": sorted({r["protocol"] for r in subset})}
    return {
        "panel_size": len(reports),
        "requested": len(panel),
        "tokens_per_record": int(n_tokens),
        "summary": summarize_fidelity(reports) if reports else {"records": 0},
        "per_round": per_round,
        "records": reports,
    }


def root_prefill_identity(
    vectors: Mapping[int, np.ndarray],
    *,
    prompt_hashes: Mapping[int, str] | None = None,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Spec §14.2 sanity check: at a byte-identical neutral root prompt the five agents'
    ``PROMPT_END`` states should be essentially identical.

    Returns the realized numbers (max absolute difference, minimum pairwise cosine,
    whether the prompts were byte-identical) and a boolean verdict against ``tolerance``.
    This is a sanity check on the capture path, **not** evidence of representational
    collapse — diversity is expected to arise only after sampled generation.
    """
    slots = sorted(vectors)
    if len(slots) < 2:
        raise ValueError("the root identity check needs at least two agents")
    X = np.stack([np.asarray(vectors[s], dtype=np.float64).reshape(-1) for s in slots])
    if len({x.shape for x in X}) > 1:  # pragma: no cover - np.stack already enforces this
        raise ValueError("root vectors have different hidden sizes")
    reference = X[0]
    diffs = np.abs(X - reference)
    norms = np.linalg.norm(X, axis=1)
    unit = X / np.where(norms > 0, norms, 1.0)[:, None]
    cos = unit @ unit.T
    identical_prompts = None
    if prompt_hashes is not None:
        identical_prompts = len({str(prompt_hashes[s]) for s in slots}) == 1
    return {
        "agents": slots,
        "n_agents": len(slots),
        "prompt_bytes_identical": identical_prompts,
        "max_abs_diff": float(diffs.max()),
        "mean_abs_diff": float(diffs.mean()),
        "min_pairwise_cosine": float(cos.min()),
        "max_relative_diff": float(diffs.max() / max(float(np.abs(reference).max()), 1e-12)),
        "tolerance": float(tolerance),
        "identical": bool(diffs.max() <= tolerance) and (identical_prompts is not False),
        "nonclaim": ("A byte-identical root prompt should give essentially identical initial "
                     "states; this is a capture sanity check, not evidence of collapse (spec 14.2)."),
    }


# --------------------------------------------------------------------------- cell worker


#: Plan §5.6 preregistered degradation, if the hour-30 capture deadline or the replay
#: GPU-hour ceiling binds first: the primary layer and anchors 1 and 3 only, so the
#: fallback is a *complete cohort at reduced depth*, never a 72% cohort.
DEGRADED_ANCHORS: tuple[str, ...] = (AA.ANCHOR_PROMPT_END, AA.ANCHOR_ANSWER_END)


def primary_block(num_layers: int) -> int:
    """``floor(L/2)-1`` — the spec §14.2 primary depth (17 for Qwen3-8B's 36 layers)."""
    from agents_scaling.study.neural.engine import blocks_for

    return blocks_for(int(num_layers), (K.CAPTURE_PRIMARY_LAYER,))[0]


def manifest_from_item_file(path: str | os.PathLike) -> list[CaptureWorkItem]:
    """Work items for one ``cells/<cell_id>/items/<item_id>.json`` file (WP4 ``ArdItemResult``).

    This *is* the decision manifest: one row per ``DecisionRef``, carrying the join keys.
    Decisions of the CAPTURE protocols only (``contract.CAPTURE_PROTOCOLS``, plus every
    round-0 root, which is shared by all of them — spec §14.2 "Reuse root captures across
    protocols").
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    split = str(payload.get("split") or "")
    family = str(payload.get("family") or "")
    item_id = str(payload.get("source_id") or payload.get("item_id") or "")
    cell = payload.get("cell") or {}
    cell_id = str(cell.get("cell_id") or "")
    captured = {p.value for p in K.CAPTURE_PROTOCOLS}
    out: list[CaptureWorkItem] = []
    for episode in payload.get("episodes") or ():
        protocol = str(episode.get("protocol") or "")
        cohort = str(episode.get("cohort") or K.Cohort.CAPTURE.value)
        for decision in episode.get("decisions") or ():
            round_ = int(decision.get("round") or 0)
            if protocol not in captured and round_ != 0:
                continue
            if not decision.get("decision_id"):
                continue        # a context failure never reached vLLM: nothing to replay
            out.append(CaptureWorkItem(
                decision_id=str(decision["decision_id"]),
                episode_id=str(decision.get("episode_id") or episode.get("episode_id") or ""),
                item_id=str(decision.get("item_id") or episode.get("item_id") or item_id),
                agent_id=int(decision.get("agent_id") or 0),
                round_=round_,
                protocol=str(decision.get("protocol") or protocol),
                cohort=cohort,
                split=split,
                family=family,
                checkpoint=str(decision.get("checkpoint") or episode.get("checkpoint") or K.CHECKPOINT_SMALL),
                boundary_injected=bool(decision.get("boundary_injected", False)),
                cell_id=cell_id,
                request_id=str(decision.get("request_id") or decision["decision_id"]),
            ))
    return out


def manifest_for_item(
    run_root: str | os.PathLike,
    item_id: str,
    *,
    cell_ids: Sequence[str] = (),
    manifest: str | os.PathLike | None = None,
) -> tuple[CaptureWorkItem, ...]:
    """The frozen work list for one item: an explicit manifest file, or the item files of
    the named generation cells (never a directory scan of the whole run root)."""
    if manifest is not None:
        return work_list([w for w in load_manifest(manifest) if str(w.item_id) == str(item_id)])
    rows: list[CaptureWorkItem] = []
    for cell_id in cell_ids:
        path = Path(run_root) / "cells" / str(cell_id) / "items" / f"{item_id}.json"
        if path.exists():
            rows.extend(manifest_from_item_file(path))
    return work_list(rows)


def default_engine_factory(options: Mapping[str, Any]) -> tuple[Any, Any]:
    """``(engine, tokenizer)`` from a local HF snapshot (offline, ``local_files_only``).

    ``blocks`` default to ``engine.blocks_for(L, contract.CAPTURE_RELATIVE_LAYERS)``, which
    is the spec's ``floor(L/4)-1`` triple — ``(8, 17, 26)`` for Qwen3-8B's 36 layers — and
    collapse to the primary depth alone under the preregistered degradation.
    """
    from transformers import AutoTokenizer

    from agents_scaling.study.neural.engine import CaptureEngine, blocks_for

    snapshot = Path(options["snapshot_path"])
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    if options.get("degraded"):
        blocks = (primary_block(num_layers),)
    else:
        blocks = blocks_for(num_layers, options.get("fracs") or K.CAPTURE_RELATIVE_LAYERS)
    engine = CaptureEngine(
        snapshot,
        blocks,
        device_map=options.get("device_map", "auto"),
        max_batch_tokens=int(options.get("max_batch_tokens", 16384)),
        max_batch_rows=options.get("max_batch_rows"),
        mask_policy=str(options.get("mask_policy", "causal")),
    )
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    return engine, tokenizer


#: Injection point: tests and the degraded path replace this without touching the worker.
ENGINE_FACTORY: Callable[[Mapping[str, Any]], tuple[Any, Any]] = default_engine_factory


def run_capture_cell(runner: Any, item_id: str) -> None:
    """CAPTURE cell worker (``ard.kinds.KIND_TABLE``): replay one item's decisions.

    No vLLM endpoint is involved — this is a teacher-forced replay on a replay GPU
    (spec §14.5) — so the cell's only inputs are the committed request records and the
    decision manifest of the generation cells named in ``options["cells"]``.
    """
    from agents_scaling.study.inference.store import RequestStore

    options = dict(runner.ard_cell.options)
    started = runner.clock()
    items = manifest_for_item(runner.run_root, item_id,
                              cell_ids=options.get("cells") or (),
                              manifest=options.get("manifest"))
    task = runner.tasks.get(item_id) if hasattr(runner.tasks, "get") else runner.tasks[item_id]
    if not items:
        if task is not None:
            runner.write_item(task, (), status="no_decisions", started=started)
        return
    engine, tokenizer = ENGINE_FACTORY(options)
    anchors_option = tuple(options.get("anchors") or (DEGRADED_ANCHORS if options.get("degraded") else K.CAPTURE_ANCHORS))
    config = CaptureConfig(
        revision=str(options.get("revision") or K.REVISIONS.get(str(options.get("checkpoint") or K.CHECKPOINT_SMALL), "")),
        checkpoint=str(options.get("checkpoint") or K.CHECKPOINT_SMALL),
        shard_name=str(options.get("shard_name") or f"ard36.{runner.ard_cell.cell_id}".replace("/", "_")),
        chunk_rows=int(options.get("chunk_rows", 128)),
        group_size=int(options.get("group_size", 8)),
        anchors=anchors_option,
        entropy_anchors=frozenset(a for a in K.ENTROPY_ANCHORS if a in anchors_option),
        strict_records=bool(options.get("strict_records", True)),
        check_identity=bool(options.get("check_identity", True)),
        allow_model_mismatch=bool(options.get("allow_model_mismatch", False)),
        fidelity_records=int(options.get("fidelity_records", 0)),
        fidelity_tokens=int(options.get("fidelity_tokens", 20)),
    )
    stats = run_capture(items, engine=engine, tokenizer=tokenizer,
                        store=RequestStore(Path(runner.run_root) / "requests"),
                        run_root=runner.run_root, config=config, staging=options.get("staging"))
    if hasattr(runner, "event"):
        runner.event("captured", captured=stats["captured"], rows=stats["rows"], tokens=stats["tokens"])
    if task is not None:
        runner.write_item(task, (), started=started)
