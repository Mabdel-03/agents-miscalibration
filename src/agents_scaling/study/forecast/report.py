"""FINAL_HANDOFF_REPORT compiler (spec §8.7; brief 09_neural_readouts_brief.md R1 step 1).

The report is compiled **after** ``SELECTIONS.json`` seals an item and **before** any
correctness join, from sealed artifacts only: the item file
``<run_root>/cells/<cell_id>/items/<source_id>.json`` (``EpisodeResult`` layout,
``study/types.py``), the seal register ``<run_root>/seals/<sha>/SELECTIONS.json``
(``study/selection/seal.py``) and the public task export.  ``data/protected`` is never read.

Frozen layout of the report text (inserted verbatim into ``prompts/templates/forecast.txt``
between the ``=== REPORT (observable state) ===`` / ``=== END REPORT ===`` lines by
``prompts.render.render_forecast``)::

    === TASK ===
    <exact public task text>                       ← task span; task_only_anchor = its end
    === END TASK ===
    --- item ---
    <compact JSON: ids, domain, format, method, N, B, checkpoint, status>
    --- selected_candidate ---
    <complete §3.5 candidate JSON | the exact §3.5 failure sentinel>
    --- selected_personal_confidence ---
    {"scope":"PERSONAL_FINAL","value":<float|null>,"missing":<bool>}
    --- decision_rule ---                          ← evidence span starts here
    <frozen declared decision rule of the method>
    --- vote_metadata ---
    <compact JSON from the sealed selection record>
    --- budget_metadata ---
    <compact JSON from the episode ledger and §3.4 counters>
    --- nonselected_candidates: k of n ---         (or "none" when the pool has no other member)
    [packet 1]
    <exact §4.3 packet bytes>
    ...                                            ← evidence span ends at the end of the text

Frozen rules
* **Selected candidate**: the sealed VOTE selection over the method's primary pool
  (``archive`` for IND_VOTE, ``latest_slots`` for DEC, ``native`` for CEN_FLAT).  A sealed
  ``selected_candidate_id`` of ``None`` (no valid complete candidate) renders the exact
  ``SENTINEL_JSON`` and sets ``personal_confidence_missing``.
* **Evidence order** (§8.7): decision rule; vote/selection metadata; budget and call/role
  counts; then ≤ ``MAX_NONSELECTED_PACKETS`` non-selected pool members in the blind order
  ``identity.blind_order_key(study_seed, "REPORT_PACKETS", source_id, candidate_id)`` —
  independent of validity, confidence and correctness (§4.1).  Invalid members yield the
  typed unavailable packet (a failed opportunity is observable evidence).
* **Evidence cap**: the *complete rendered* evidence span (decision rule → last packet, one
  contiguous byte span of the report) is tokenized as a whole and must be ≤
  ``EVIDENCE_TOKENS_CAP`` (8,192) recipient tokens.  Metadata and wrappers are reserved
  first (the zero-packet render must fit, else ``ProtocolError``); packets are admitted in
  order while the whole span fits; the first packet that does not fit whole is re-compiled
  under the largest packet cap (binary search, ``packets.compile_packet``'s deterministic
  excerpt clipper) that fits, or dropped when even its skeleton does not fit; no later
  packet is admitted (it is the *last* packet, §8.7).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents_scaling.study import identity
from agents_scaling.study.config import StudyConfig, load_config
from agents_scaling.study.packets import compile_packet, make_counter
from agents_scaling.study.prompts import render as R
from agents_scaling.study.selection import seal as seals
from agents_scaling.study.types import (
    PACKET_FINAL_TOKENS_CAP,
    PACKET_TOKENS_CAP,
    SELECTOR_VOTE,
    SENTINEL_JSON,
    Candidate,
    CandidateRecord,
    CellSpec,
    Method,
    Packet,
    ProtocolError,
    PublicTask,
)

REPORT_SCHEMA_VERSION = 1
#: §8.7 evidence cap (recipient tokens of the complete rendered evidence span).
EVIDENCE_TOKENS_CAP = 8192
#: §8.7 "up to four nonselected evidence artifacts".
MAX_NONSELECTED_PACKETS = 4
#: HMAC namespace of the blind packet order (brief R1 step 1).
REPORT_PACKETS_NAMESPACE = "REPORT_PACKETS"
#: Methods with a FINAL_HANDOFF_REPORT node (CEN_RLM is not implemented in this run).
REPORT_METHODS: tuple[Method, ...] = (Method.IND_VOTE, Method.DEC, Method.CEN_FLAT)
#: The policy's own primary pool kind (seal.py: ``archive`` / ``latest_slots`` / ``native``).
PRIMARY_POOL_KIND: Mapping[Method, str] = {
    Method.IND_VOTE: seals.POOL_ARCHIVE,
    Method.DEC: seals.POOL_LATEST,
    Method.CEN_FLAT: seals.POOL_NATIVE,
}
#: Frozen declared decision rules (§4.4; the reader sees exactly these bytes).
DECISION_RULES: Mapping[Method, str] = {
    Method.IND_VOTE: (
        "VOTE: plurality over the normalized final answers of every completed independent attempt "
        "(archive pool); ties are broken by a blind study-hashed key; attempts without a valid "
        "complete candidate remain failed opportunities and cannot win."
    ),
    Method.DEC: (
        "VOTE: plurality over the normalized final answers of the N latest member outputs after the "
        "completed revision rounds (latest_slots pool); ties are broken by a blind study-hashed key; "
        "invalid latest outputs remain failed opportunities and cannot win."
    ),
    Method.CEN_FLAT: (
        "NATIVE: the coordinator's own final complete candidate is the episode output (synthesis over "
        "the returned worker results; no vote); an invalid final coordinator action is a failed episode."
    ),
}
#: Per-method counters copied into the vote metadata when the episode recorded them.
_METHOD_COUNTERS: Mapping[Method, tuple[str, ...]] = {
    Method.IND_VOTE: ("optional_draws",),
    Method.DEC: ("rounds_completed", "r_max", "r_max_binding"),
    Method.CEN_FLAT: ("delegation_cycles", "hub_calls", "realized_participation", "workers_assigned", "hub_action_errors"),
}
_BUDGET_COUNTERS: tuple[str, ...] = (
    "calls_by_role",
    "context_failures_by_role",
    "model_invocations",
    "opportunities",
    "complete_candidates",
    "candidate_opportunities",
    "assigned_roster",
    "unique_actors_used",
)

SECTION_ITEM = "item"
SECTION_SELECTED = "selected_candidate"
SECTION_CONFIDENCE = "selected_personal_confidence"
SECTION_RULE = "decision_rule"
SECTION_VOTE = "vote_metadata"
SECTION_BUDGET = "budget_metadata"
SECTION_PACKETS = "nonselected_candidates"
PACKET_ITEM_LABEL = "packet"
PERSONAL_SCOPE = "PERSONAL_FINAL"

Counter = Callable[[str], int]


def _blen(text: str) -> int:
    return len(text.encode("utf-8"))


def _header(name: str) -> str:
    return R.FIELD_OPEN + name + R.FIELD_CLOSE + "\n"


# --------------------------------------------------------------------------- result object


@dataclass(frozen=True)
class Report:
    """One compiled FINAL_HANDOFF_REPORT (immutable; ``text`` is what the reader sees)."""

    schema_version: int
    report_id: str
    source_id: str
    cell_id: str
    method: Method
    seal: str
    selection_id: str
    pool_id: str
    item: dict[str, Any]
    task_text: str
    selected_candidate_id: str | None
    selected_candidate: Candidate | None
    selected_personal_confidence: float | None
    personal_confidence_missing: bool
    decision_rule: str
    vote_metadata: dict[str, Any]
    budget_metadata: dict[str, Any]
    packets: tuple[Packet, ...]
    nonselected: tuple[dict[str, Any], ...]
    nonselected_total: int
    packets_clipped_last: bool
    packets_dropped: int
    text: str
    spans: dict[str, Any]
    task_only_anchor: int
    evidence_tokens: int

    @property
    def selected_is_sentinel(self) -> bool:
        return self.selected_candidate is None

    @property
    def text_sha256(self) -> str:
        return identity.sha256_hex(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "source_id": self.source_id,
            "cell_id": self.cell_id,
            "method": self.method.value,
            "seal": self.seal,
            "selection_id": self.selection_id,
            "pool_id": self.pool_id,
            "item": dict(self.item),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate": None if self.selected_candidate is None else self.selected_candidate.to_dict(),
            "selected_is_sentinel": self.selected_is_sentinel,
            "selected_personal_confidence": self.selected_personal_confidence,
            "personal_confidence_missing": self.personal_confidence_missing,
            "decision_rule": self.decision_rule,
            "vote_metadata": dict(self.vote_metadata),
            "budget_metadata": dict(self.budget_metadata),
            "nonselected": [dict(n) for n in self.nonselected],
            "nonselected_total": self.nonselected_total,
            "packets_clipped_last": self.packets_clipped_last,
            "packets_dropped": self.packets_dropped,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "spans": dict(self.spans),
            "task_only_anchor": self.task_only_anchor,
            "evidence_tokens": self.evidence_tokens,
            "evidence_tokens_cap": EVIDENCE_TOKENS_CAP,
        }


# --------------------------------------------------------------------------- sealed inputs


def find_sealed_selection(selections: Mapping[str, Any], source_id: str, cell_id: str, method: Method) -> dict[str, Any]:
    """The sealed VOTE selection over ``method``'s primary pool of ``(cell_id, source_id)``.

    Raises ``ProtocolError`` when the item is not sealed (compilation refuses, §3.6/P0-3)
    or when the register is ambiguous.
    """
    if method not in REPORT_METHODS:
        raise ProtocolError(f"{method.value} has no FINAL_HANDOFF_REPORT node (supported: {[m.value for m in REPORT_METHODS]})")
    kind = PRIMARY_POOL_KIND[method]
    found = [
        s
        for s in seals.selections_of_item(selections, source_id)
        if s.get("cell_id") == cell_id and s.get("selector_id") == SELECTOR_VOTE and s.get("pool_kind") == kind
    ]
    if not found:
        raise ProtocolError(
            f"{cell_id}/{source_id}: no sealed {SELECTOR_VOTE} selection over the {kind} pool; refusing to compile a report before the seal"
        )
    if len(found) > 1:
        raise ProtocolError(f"{cell_id}/{source_id}: {len(found)} sealed {kind} selections (register ambiguity)")
    return found[0]


def blind_packet_order(study_seed: bytes, source_id: str, records: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    """§8.7 study-hashed order of the non-selected artifacts (blind to validity/confidence)."""
    return sorted(records, key=lambda r: identity.blind_order_key(study_seed, REPORT_PACKETS_NAMESPACE, source_id, r.candidate_id))


def _load_task(run_root: Path, source_id: str, tasks: Mapping[str, PublicTask] | None) -> PublicTask:
    if tasks is None:
        from agents_scaling.study.data.public import load_public_tasks  # public view only (§10.6)

        tasks = {t.source_id: t for t in load_public_tasks(run_root)}
    task = tasks.get(source_id)
    if task is None:
        raise ProtocolError(f"{source_id} is not in the public export")
    if not task.task_text:
        raise ProtocolError(f"{source_id}: empty task text")
    return task


# --------------------------------------------------------------------------- metadata blocks


def _item_block(cell: CellSpec, item: Mapping[str, Any], task: PublicTask) -> dict[str, Any]:
    return {
        "source_id": task.source_id,
        "domain": task.domain.value,
        "split": task.split,
        "answer_format": task.answer_format,
        "method": cell.method.value,
        "checkpoint": cell.checkpoint,
        "N": int(cell.N),
        "B": int(cell.B),
        "framing": cell.framing.value,
        "episode_rep": int(cell.episode_rep),
        "status": item.get("status"),
    }


def _vote_metadata(method: Method, sealed: Mapping[str, Any], counters: Mapping[str, Any]) -> dict[str, Any]:
    rec = sealed["record"]
    out: dict[str, Any] = {
        "selector_id": sealed["selector_id"],
        "pool_kind": sealed["pool_kind"],
        "pool_size": len(rec["pool_candidate_ids"]),
        "planned_count": rec.get("planned_count"),
        "valid_count": rec.get("valid_count"),
        "no_valid_candidate": bool(rec.get("no_valid_candidate")),
        "winning_count": rec.get("winning_count"),
        "tied_classes": rec.get("tied_classes"),
        "all_singleton": bool(rec.get("all_singleton")),
        "grouping_mode": rec.get("grouping_mode"),
        "grouping_mode_counts": dict(rec.get("grouping_mode_counts") or {}),
    }
    for key in _METHOD_COUNTERS.get(method, ()):
        if key in counters:
            out[key] = counters[key]
    return out


def _budget_metadata(cell: CellSpec, episode: Mapping[str, Any]) -> dict[str, Any]:
    ledger = episode.get("ledger") or {}
    counters = episode.get("counters") or {}
    out: dict[str, Any] = {
        "B": int(cell.B),
        "B_flops": ledger.get("B_flops"),
        "spent_flops": ledger.get("spent"),
        "slack_flops": ledger.get("slack"),
        "calls_admitted": ledger.get("calls_admitted"),
        "solver_call_cap": ledger.get("solver_call_cap"),
    }
    for key in _BUDGET_COUNTERS:
        if key in counters:
            out[key] = counters[key]
    out["stop_reason"] = episode.get("stop_reason")
    return out


# --------------------------------------------------------------------------- rendering


def _render(
    task_text: str,
    item_json: str,
    selected_text: str,
    confidence_json: str,
    rule: str,
    vote_json: str,
    budget_json: str,
    packets: Sequence[Packet],
    nonselected_total: int,
) -> tuple[str, dict[str, Any], int]:
    """Assemble the frozen layout → ``(text, byte spans, task_only_anchor)``."""
    pieces: list[str] = []
    spans: dict[str, Any] = {}
    offset = 0

    def emit(text: str, key: str | None = None) -> None:
        nonlocal offset
        if key is not None:
            spans[key] = (offset, offset + _blen(text))
        pieces.append(text)
        offset += _blen(text)

    emit(R.TASK_OPEN + "\n")
    emit(task_text, "task")
    task_only_anchor = offset
    emit("\n" + R.TASK_CLOSE + "\n")
    emit(_header(SECTION_ITEM))
    emit(item_json, SECTION_ITEM)
    emit("\n" + _header(SECTION_SELECTED))
    emit(selected_text, SECTION_SELECTED)
    emit("\n" + _header(SECTION_CONFIDENCE))
    emit(confidence_json, SECTION_CONFIDENCE)
    emit("\n")
    evidence_start = offset
    emit(_header(SECTION_RULE))
    emit(rule, SECTION_RULE)
    emit("\n" + _header(SECTION_VOTE))
    emit(vote_json, SECTION_VOTE)
    emit("\n" + _header(SECTION_BUDGET))
    emit(budget_json, SECTION_BUDGET)
    emit("\n")
    count = "none" if nonselected_total == 0 else f"{len(packets)} of {nonselected_total}"
    emit(_header(f"{SECTION_PACKETS}: {count}"))
    packet_spans: list[tuple[int, int]] = []
    for index, packet in enumerate(packets, 1):
        emit(f"[{PACKET_ITEM_LABEL} {index}]\n")
        start = offset
        emit(R.packet_text(packet))
        packet_spans.append((start, offset))
        if index < len(packets):
            emit("\n")
    text = "".join(pieces).rstrip("\n")
    spans["packets"] = packet_spans
    spans["evidence"] = (evidence_start, _blen(text))
    return text, spans, task_only_anchor


def _admit_packets(
    order: Sequence[CandidateRecord],
    count: Counter,
    render: Callable[[Sequence[Packet]], tuple[str, dict[str, Any], int]],
    *,
    cap: int,
    packet_cap: int,
    final_cap: int,
) -> tuple[list[Packet], list[dict[str, Any]], bool, int, int]:
    """Admit packets in order under ``cap`` tokens of the complete evidence span.

    Returns ``(packets, per-packet metadata, clipped_last, dropped, evidence_tokens)``.
    """

    def evidence_tokens(packets: Sequence[Packet]) -> int:
        text, spans, _ = render(packets)
        start, end = spans["evidence"]
        return count(text.encode("utf-8")[start:end].decode("utf-8"))

    base = evidence_tokens([])
    if base > cap:
        raise ProtocolError(f"report metadata alone renders {base} evidence tokens > cap {cap} (harness configuration defect)")
    admitted: list[Packet] = []
    metas: list[dict[str, Any]] = []
    clipped_last = False
    dropped = 0
    n = base

    def compiled(record: CandidateRecord, c: int) -> Packet | None:
        try:
            return compile_packet(record, count, int(record.slot), c, min(final_cap, c))
        except ProtocolError:
            return None  # the skeleton itself does not fit under c

    for record in order[:MAX_NONSELECTED_PACKETS]:
        packet = compiled(record, packet_cap)
        assert packet is not None  # the frozen cap always admits the skeleton
        trial = evidence_tokens(admitted + [packet])
        if trial <= cap:
            admitted.append(packet)
            metas.append(_packet_meta(record, packet, packet_cap, clipped=False))
            n = trial
            continue
        # the last packet: largest packet cap in [1, packet_cap) whose complete render fits
        lo, hi = 0, packet_cap  # invariant: lo fits (0 = absent), hi does not
        best: Packet | None = None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            candidate = compiled(record, mid)
            if candidate is not None and evidence_tokens(admitted + [candidate]) <= cap:
                lo, best = mid, candidate
            else:
                hi = mid
        if best is None:
            dropped = 1
        else:
            admitted.append(best)
            metas.append(_packet_meta(record, best, lo, clipped=True))
            clipped_last = True
            n = evidence_tokens(admitted)
        break
    if n > cap:  # BPE non-monotonicity guard: the final count is asserted, never assumed
        raise ProtocolError(f"evidence span renders {n} tokens > cap {cap} after admission")
    return admitted, metas, clipped_last, dropped, n


def _packet_meta(record: CandidateRecord, packet: Packet, cap_used: int, *, clipped: bool) -> dict[str, Any]:
    return {
        "candidate_id": record.candidate_id,
        "sender_slot": packet.sender_slot,
        "stage": record.stage,
        "valid": bool(record.valid and record.candidate is not None),
        "packet_id": packet.packet_id,
        "bytes_sha256": packet.bytes_sha256,
        "recipient_tokens": packet.recipient_tokens,
        "packet_cap_used": int(cap_used),
        "clipped_to_fit": bool(clipped),
        "unavailable": bool(packet.unavailable),
        "final_partial": bool(packet.final_partial),
        "truncated": dict(packet.truncated),
    }


# --------------------------------------------------------------------------- entry point


def compile_report(
    run_root: str | os.PathLike,
    source_id: str,
    method_cell_id: str,
    tokenizer: Any,
    *,
    seal: str,
    cfg: StudyConfig | None = None,
    selections: Mapping[str, Any] | None = None,
    tasks: Mapping[str, PublicTask] | None = None,
    evidence_cap: int = EVIDENCE_TOKENS_CAP,
    packet_cap: int = PACKET_TOKENS_CAP,
    packet_final_cap: int = PACKET_FINAL_TOKENS_CAP,
) -> Report:
    """Compile the FINAL_HANDOFF_REPORT of ``(method_cell_id, source_id)`` under ``seal``.

    ``tokenizer`` is the pinned reader tokenizer (or the test stub / a ``count`` callable);
    ``selections`` and ``tasks`` may be passed to avoid re-reading the registers per item.
    Raises ``ProtocolError`` when the item file is missing/corrupt, the selection is not
    sealed, the pool references unknown candidates, or the metadata alone exceeds the cap.
    """
    run_root = Path(run_root)
    cfg = cfg or load_config()
    count = make_counter(tokenizer)
    if selections is None:
        selections = seals.load_selections(run_root, seal)
    item = seals.load_item_file(run_root, method_cell_id, source_id)
    if item is None:
        raise ProtocolError(f"item file {seals.item_path(run_root, method_cell_id, source_id)} is missing")
    episode = item.get("episode")
    if not isinstance(episode, dict):
        raise ProtocolError(f"{method_cell_id}/{source_id}: not a generate-cell episode (no episode block)")
    cell = CellSpec.from_dict(item["cell"])
    method = Method(cell.method)
    sealed = find_sealed_selection(selections, source_id, method_cell_id, method)
    rec = sealed["record"]
    task = _load_task(run_root, source_id, tasks)

    by_id = {r.candidate_id: r for r in seals.candidate_records_of(item)}
    pool_ids = list(rec["pool_candidate_ids"])
    missing = [c for c in pool_ids if c not in by_id]
    if missing:
        raise ProtocolError(f"{method_cell_id}/{source_id}: sealed pool references unknown candidates {missing[:3]}")
    selected_id = rec.get("selected_candidate_id")
    selected: Candidate | None = None
    if selected_id is not None:
        if selected_id not in pool_ids:
            raise ProtocolError(f"{method_cell_id}/{source_id}: sealed winner {selected_id[:12]} is outside its pool")
        winner = by_id[selected_id]
        if not winner.valid or winner.candidate is None:
            raise ProtocolError(f"{method_cell_id}/{source_id}: sealed winner {selected_id[:12]} is not a valid complete candidate")
        selected = winner.candidate
    confidence = None if selected is None else float(selected.confidence)
    missing_conf = selected is None

    nonselected_records = blind_packet_order(cfg.study_seed, source_id, [by_id[c] for c in pool_ids if c != selected_id])
    counters = episode.get("counters") or {}
    item_block = _item_block(cell, item, task)
    vote_meta = _vote_metadata(method, sealed, counters)
    budget_meta = _budget_metadata(cell, episode)
    rule = DECISION_RULES[method]
    item_json = R.compact_json(item_block)
    selected_text = SENTINEL_JSON if selected is None else R.compact_json(selected.to_dict())
    confidence_json = R.compact_json({"scope": PERSONAL_SCOPE, "value": confidence, "missing": missing_conf})
    vote_json = R.compact_json(vote_meta)
    budget_json = R.compact_json(budget_meta)
    total = len(nonselected_records)

    def render(packets: Sequence[Packet]) -> tuple[str, dict[str, Any], int]:
        return _render(task.task_text, item_json, selected_text, confidence_json, rule, vote_json, budget_json, packets, total)

    packets, metas, clipped_last, dropped, evidence_tokens = _admit_packets(
        nonselected_records, count, render, cap=evidence_cap, packet_cap=packet_cap, final_cap=packet_final_cap
    )
    text, spans, task_only_anchor = render(packets)
    report_id = identity.sha256_hex(
        identity.jcs(["final_handoff_report", seal, sealed["selection_id"], method_cell_id, source_id, identity.sha256_hex(text)])
    )
    return Report(
        schema_version=REPORT_SCHEMA_VERSION,
        report_id=report_id,
        source_id=source_id,
        cell_id=method_cell_id,
        method=method,
        seal=seal,
        selection_id=str(sealed["selection_id"]),
        pool_id=str(sealed["pool_id"]),
        item=item_block,
        task_text=task.task_text,
        selected_candidate_id=selected_id,
        selected_candidate=selected,
        selected_personal_confidence=confidence,
        personal_confidence_missing=missing_conf,
        decision_rule=rule,
        vote_metadata=vote_meta,
        budget_metadata=budget_meta,
        packets=tuple(packets),
        nonselected=tuple(metas),
        nonselected_total=total,
        packets_clipped_last=clipped_last,
        packets_dropped=dropped,
        text=text,
        spans=spans,
        task_only_anchor=task_only_anchor,
        evidence_tokens=evidence_tokens,
    )


__all__ = [
    "DECISION_RULES",
    "EVIDENCE_TOKENS_CAP",
    "MAX_NONSELECTED_PACKETS",
    "PRIMARY_POOL_KIND",
    "REPORT_METHODS",
    "REPORT_PACKETS_NAMESPACE",
    "REPORT_SCHEMA_VERSION",
    "Report",
    "blind_packet_order",
    "compile_report",
    "find_sealed_selection",
]
