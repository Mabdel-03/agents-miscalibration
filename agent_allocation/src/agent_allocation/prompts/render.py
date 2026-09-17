"""Frozen prompt templates, their hash lock, and request rendering (agent 03; MASTER_SPEC §4.3, §5, §8).

Templates are the pack files in ``<project>/prompts/`` (one source of truth, pinned by the pack's CHECKSUMS.json):
``system.txt`` is every request's system message; the user message is

* ROOT, DEEP      -> ``root.txt``    (task + output requirement)
* PRIVATE, CP     -> ``private.txt`` (``{own_packet}`` = the single parent's latest visible solution)
* PEER            -> ``peer.txt``    (``{own_packet}`` = the receiver's own previous candidate, same root_slot;
                                      ``{peer_packets}`` = the other n-1 parents in a blind permutation)
* POOLED          -> ``pooled.txt``  (``{candidate_packets}`` = the four roots in a blind permutation)

Packet rules:

* A packet carries ``Candidate.visible_content`` only (outer whitespace stripped) -- never reasoning, parse fields,
  scores, grades or history.
* ``None`` parent, or a candidate with no visible text, renders exactly ``[No prior candidate response is
  available.]``.
* A candidate whose ``parse_status`` is not VALID but that has visible text is relayed as bounded raw text under
  the format-neutral marker ``[A prior candidate response is available as raw visible text.]``.
* Envelope: the relayed text of one packet is at most :data:`PACKET_ENVELOPE_CHARS` characters (the omission
  marker counts; the availability marker and the ``CANDIDATE i`` label are fixed overhead outside it). Longer text
  keeps every complete ``<final>...</final>`` element intact and in place and trims only the text outside the
  final elements symmetrically: of the K characters outside the finals that fit, the first ceil(K/2) and the last
  floor(K/2) survive and the middle is replaced by :data:`OMISSION_MARKER` (one marker per contiguous cut).
  Characters and UTF-8 bytes removed are reported. If a VALID candidate's final element plus the omission
  marker(s) cannot fit, :class:`ContextLimitEvent` is raised for the lead to record. A raw relay (not VALID) is always bounded: it keeps
  its complete final elements when they fit and otherwise trims the whole raw text symmetrically under the marker.
* Peer and pooled order: sort source slots by ``canonical_hash([namespace, seed, item_id, n, round, receiver_slot,
  source_slot])`` -- a stable, correctness-blind function of (seed, item, n, round, receiver); never return order.
  Labels are neutral: ``CANDIDATE 1``, ``CANDIDATE 2``, ...
* Nothing about horizon, round, role, slot or team size is written into a prompt.

``prompt_version()`` is ``"prompts-v1-"`` + the first 12 hex of ``canonical_hash(prompt_hashes())``; rendering refuses
a spec whose ``prompt_version`` differs (the templates it was frozen under are not the ones on disk).

Config keys consumed (R9): ``seed`` arrives as the ``seed`` argument (config ``seed``); nothing else.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_allocation.contracts import (
    Candidate,
    GenerationKind,
    GenerationSpec,
    OutputKind,
    ParseStatus,
    PublicTask,
    canonical_hash,
    canonical_json,
    public_task_hash,
)
from agent_allocation.paths import PROJECT_ROOT

TEMPLATE_DIR: Path = PROJECT_ROOT / "prompts"
TEMPLATE_FILES: tuple[str, ...] = (
    "system.txt", "root.txt", "private.txt", "peer.txt", "pooled.txt", "output_requirements.json",
)
PROMPT_VERSION_PREFIX = "prompts-v1-"

#: Characters of relayed candidate text per packet (see the module docstring for the trimming rule). 12,000 chars
#: covers a whole completion at the main-A code cap (A + 2,048 = 3,072 tokens at <= ~4 chars/token), so at A=1,024 a
#: VALID final can never raise ContextLimitEvent and trimming touches only runaway rationales; a five-agent PEER
#: user message stays below ~60k chars (~15k tokens). Freeze (or tighten from pilot data) before main dispatch.
PACKET_ENVELOPE_CHARS = 12000

ABSENT_MARKER = "[No prior candidate response is available.]"
RAW_MARKER = "[A prior candidate response is available as raw visible text.]"
OMISSION_MARKER = "[... text omitted for length ...]"
CANDIDATE_LABEL = "CANDIDATE {index}"
PEER_ORDER_NAMESPACE = "peer_order_v1"
POOLED_ORDER_NAMESPACE = "pooled_order_v1"

TEMPLATE_FOR_KIND: Mapping[GenerationKind, str] = {
    GenerationKind.ROOT: "root.txt",
    GenerationKind.DEEP: "root.txt",
    GenerationKind.PRIVATE: "private.txt",
    GenerationKind.CP: "private.txt",
    GenerationKind.PEER: "peer.txt",
    GenerationKind.POOLED: "pooled.txt",
}
PLACEHOLDERS: Mapping[str, frozenset[str]] = {
    "root.txt": frozenset({"task_prompt", "output_requirement"}),
    "private.txt": frozenset({"task_prompt", "output_requirement", "own_packet"}),
    "peer.txt": frozenset({"task_prompt", "output_requirement", "own_packet", "peer_packets"}),
    "pooled.txt": frozenset({"task_prompt", "output_requirement", "candidate_packets"}),
}
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_FINAL_OPEN = re.compile(r"<final\b[^>]{0,64}>", re.IGNORECASE)
_FINAL_CLOSE = re.compile(r"</final\s*>", re.IGNORECASE)
_OMISSION_INSERT = f"\n{OMISSION_MARKER}\n"
_TASK_KIND = {"math500": OutputKind.MATH, "mmlu_pro": OutputKind.OPTION, "mbpp_plus": OutputKind.PYTHON}


class PromptError(ValueError):
    """A render request that does not match its spec, parents, task or the frozen templates."""


class ContextLimitEvent(RuntimeError):
    """A packet whose ``<final>`` element(s) alone exceed the envelope: record it and apply the failure policy."""

    def __init__(self, *, source_node_id: str | None, final_chars: int, visible_chars: int,
                 envelope_chars: int, receiver_node_id: str | None = None) -> None:
        self.record: dict[str, Any] = {
            "event": "context_limit",
            "receiver_node_id": receiver_node_id,
            "source_node_id": source_node_id,
            "final_chars": final_chars,
            "visible_chars": visible_chars,
            "envelope_chars": envelope_chars,
        }
        super().__init__(f"packet from {source_node_id}: final element(s) of {final_chars} chars cannot fit the "
                         f"{envelope_chars}-char envelope (visible text {visible_chars} chars)")


# ---------------------------------------------------------------------------------------------------------
# Templates and the hash lock
# ---------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _template_bytes() -> dict[str, bytes]:
    missing = [name for name in TEMPLATE_FILES if not (TEMPLATE_DIR / name).is_file()]
    if missing:
        raise PromptError(f"prompt templates missing from {TEMPLATE_DIR}: {missing}")
    return {name: (TEMPLATE_DIR / name).read_bytes() for name in TEMPLATE_FILES}


def _template_text(name: str) -> str:
    return _template_bytes()[name].decode("utf-8")


def prompt_hashes() -> dict[str, str]:
    """``{template file name: sha256 of its bytes}`` for the whole frozen template set."""
    return {name: sha256(data).hexdigest() for name, data in _template_bytes().items()}


def prompt_version() -> str:
    return PROMPT_VERSION_PREFIX + canonical_hash(prompt_hashes())[:12]


def prompt_lock_record() -> dict[str, Any]:
    """What the lead writes to ``paths.prompt_lock``; :func:`verify_prompt_lock` checks it later."""
    return {"prompt_version": prompt_version(), "templates": prompt_hashes(),
            "packet_envelope_chars": PACKET_ENVELOPE_CHARS}


def verify_prompt_lock(record: Mapping[str, Any]) -> None:
    current = prompt_lock_record()
    drift = {key: (record.get(key), value) for key, value in current.items() if record.get(key) != value}
    if drift:
        raise PromptError(f"prompt lock drift (locked, on disk): {drift}")


@lru_cache(maxsize=1)
def _requirements() -> dict[str, str]:
    return json.loads(_template_text("output_requirements.json"))


def output_requirement(task: PublicTask) -> str:
    """The benchmark's output requirement; BBH picks ``bbh_boolean`` or ``bbh_choice`` by output kind."""
    if task.benchmark == "bbh_logic4":
        if task.output_kind is OutputKind.BOOLEAN:
            key = "bbh_boolean"
        elif task.output_kind is OutputKind.OPTION:
            key = "bbh_choice"
        else:
            raise PromptError(f"{task.item_id}: BBH output kind {task.output_kind.value!r} has no requirement")
        if task.subtask is not None and (task.subtask == "boolean_expressions") != (key == "bbh_boolean"):
            raise PromptError(f"{task.item_id}: subtask {task.subtask!r} disagrees with output kind "
                              f"{task.output_kind.value!r}")
        return _requirements()[key]
    expected = _TASK_KIND.get(task.benchmark)
    if expected is None:
        raise PromptError(f"{task.item_id}: unknown benchmark {task.benchmark!r}")
    if task.output_kind is not expected:
        raise PromptError(f"{task.item_id}: {task.benchmark} expects output kind {expected.value!r}, "
                          f"got {task.output_kind.value!r}")
    return _requirements()[task.benchmark]


def _substitute(template_name: str, values: Mapping[str, str]) -> str:
    """Replace the template's declared placeholders once; substituted text is never rescanned (tasks have braces)."""
    template = _template_text(template_name)
    declared = set(_PLACEHOLDER.findall(template))
    if declared != set(PLACEHOLDERS[template_name]) or declared != set(values):
        raise PromptError(f"{template_name}: placeholders {sorted(declared)} != expected "
                          f"{sorted(PLACEHOLDERS[template_name])} / supplied {sorted(values)}")
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


# ---------------------------------------------------------------------------------------------------------
# Packets
# ---------------------------------------------------------------------------------------------------------

def _utf8(text: str) -> bytes:
    """UTF-8 bytes; a lone surrogate decoded from a response's JSON escapes must not crash rendering."""
    return text.encode("utf-8", "surrogatepass")


def _audit_hash(obj: Any) -> str:
    """``canonical_hash`` (identical for every well-formed string), tolerant of lone surrogates."""
    return sha256(_utf8(canonical_json(obj))).hexdigest()


def _final_spans(text: str) -> list[tuple[int, int]]:
    """Complete ``<final>...</final>`` elements, leftmost-first and non-overlapping, in linear time.

    Same spans as ``re.finditer(r"<final\\b[^>]{0,64}>.*?</final\\s*>", IGNORECASE | DOTALL)`` without its
    quadratic backtracking on degenerate outputs full of unclosed ``<final>`` tags.
    """
    closes = [(m.start(), m.end()) for m in _FINAL_CLOSE.finditer(text)]
    spans: list[tuple[int, int]] = []
    ci = 0
    cursor = 0
    for opening in _FINAL_OPEN.finditer(text):
        if opening.start() < cursor:
            continue
        while ci < len(closes) and closes[ci][0] < opening.end():
            ci += 1
        if ci == len(closes):
            break
        spans.append((opening.start(), closes[ci][1]))
        cursor = closes[ci][1]
        ci += 1
    return spans


def _cuts_for(text: str, spans: Sequence[tuple[int, int]], keep: int) -> list[tuple[int, int]]:
    """Cut ranges (text positions) that remove the middle of the text outside ``spans``, keeping ``keep`` chars."""
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < len(text):
        gaps.append((cursor, len(text)))
    outside = sum(b - a for a, b in gaps)
    head = (keep + 1) // 2
    cut_lo, cut_hi = head, outside - (keep - head)          # virtual coordinates over the outside text
    cuts: list[tuple[int, int]] = []
    offset = 0
    for a, b in gaps:
        lo, hi = max(cut_lo, offset), min(cut_hi, offset + (b - a))
        if lo < hi:
            cuts.append((a + lo - offset, a + hi - offset))
        offset += b - a
    return cuts


def _plan_cuts(text: str, spans: Sequence[tuple[int, int]], envelope: int) -> list[tuple[int, int]] | None:
    """Cuts that fit ``text`` into ``envelope`` keeping ``spans`` intact, or None if the spans alone cannot fit."""
    protected = sum(b - a for a, b in spans)
    n_cuts = 1
    while True:
        keep = envelope - protected - n_cuts * len(_OMISSION_INSERT)
        if keep < 0:
            return None
        cuts = _cuts_for(text, spans, keep)
        if len(cuts) <= n_cuts:
            return cuts
        n_cuts = len(cuts)


def bound_packet_text(text: str, *, envelope: int = PACKET_ENVELOPE_CHARS, source_node_id: str | None = None,
                      strict_final: bool = True) -> tuple[str, dict[str, Any]]:
    """Fit ``text`` into ``envelope`` characters without cutting a complete ``<final>`` element.

    Returns ``(bounded_text, trim_report)``. If the final element(s) alone (plus the omission marker) do not fit:
    ``strict_final`` (VALID candidates) raises :class:`ContextLimitEvent`; otherwise (raw relays) the whole text is
    trimmed symmetrically and ``finals_protected`` is reported False.
    """
    if envelope <= 2 * len(_OMISSION_INSERT):
        raise ValueError(f"envelope of {envelope} chars cannot hold an omission marker")
    spans = _final_spans(text)     # counted for every packet, trimmed or not (the report is audit data)
    report: dict[str, Any] = {"original_chars": len(text), "original_bytes": len(_utf8(text)),
                              "trimmed": False, "cuts": 0, "chars_removed": 0, "bytes_removed": 0,
                              "final_elements": len(spans), "finals_protected": True}
    if len(text) <= envelope:
        report["relayed_chars"] = len(text)
        return text, report
    cuts = _plan_cuts(text, spans, envelope)
    if cuts is None:
        if strict_final:
            raise ContextLimitEvent(source_node_id=source_node_id, final_chars=sum(b - a for a, b in spans),
                                    visible_chars=len(text), envelope_chars=envelope)
        spans = []
        report["finals_protected"] = False
        cuts = _plan_cuts(text, spans, envelope)
        assert cuts is not None
    pieces: list[str] = []
    cursor = 0
    for a, b in cuts:
        pieces.append(text[cursor:a])
        pieces.append(_OMISSION_INSERT)
        cursor = b
    pieces.append(text[cursor:])
    bounded = "".join(pieces)
    if len(bounded) > envelope:  # pragma: no cover - arithmetic guard
        raise AssertionError(f"packet bound produced {len(bounded)} > {envelope} chars")
    if any(text[a:b] not in bounded for a, b in spans):  # pragma: no cover - cuts never intersect finals
        raise AssertionError("packet bound altered a <final> element")
    report.update(trimmed=True, cuts=len(cuts), chars_removed=sum(b - a for a, b in cuts),
                  bytes_removed=sum(len(_utf8(text[a:b])) for a, b in cuts), relayed_chars=len(bounded))
    return bounded, report


def render_packet(candidate: Candidate | None, *, source_node_id: str | None = None,
                  envelope: int = PACKET_ENVELOPE_CHARS) -> tuple[str, dict[str, Any]]:
    """One packet body (no label) and its report entry."""
    content = candidate.visible_content.strip() if candidate is not None and candidate.visible_content else ""
    entry: dict[str, Any] = {
        "source_node_id": source_node_id,
        "parse_status": candidate.parse_status.value if candidate is not None else None,
    }
    if not content:
        entry.update(availability="absent", original_chars=0, original_bytes=0, relayed_chars=0, trimmed=False,
                     cuts=0, chars_removed=0, bytes_removed=0, final_elements=0, finals_protected=True)
        return ABSENT_MARKER, entry
    valid = candidate.parse_status is ParseStatus.VALID
    bounded, trim = bound_packet_text(content, envelope=envelope, source_node_id=source_node_id, strict_final=valid)
    entry.update(trim)
    if valid:
        entry["availability"] = "valid"
        return bounded, entry
    entry["availability"] = "raw"
    return f"{RAW_MARKER}\n{bounded}", entry


def blind_order(namespace: str, *, seed: int, item_id: str, team_size: int, update_round: int, receiver_slot: int,
                source_slots: Sequence[int]) -> tuple[int, ...]:
    """Correctness-blind display order of ``source_slots``: a stable hash of (seed, item, n, round, receiver)."""
    def key(slot: int) -> tuple[str, int]:
        return canonical_hash([namespace, seed, item_id, team_size, update_round, receiver_slot, slot]), slot
    return tuple(sorted(source_slots, key=key))


# ---------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------

def _check_request(spec: GenerationSpec, task: PublicTask,
                   parents: Sequence[tuple[GenerationSpec, Candidate | None]]) -> list[tuple[GenerationSpec, Candidate | None]]:
    if spec.item_id != task.item_id or spec.benchmark != task.benchmark:
        raise PromptError(f"spec {spec.node_id} is for {spec.benchmark}:{spec.item_id}, task is "
                          f"{task.benchmark}:{task.item_id}")
    if spec.input_manifest_hash != public_task_hash(task):
        raise PromptError(f"spec {spec.node_id}: input_manifest_hash does not match this PublicTask")
    if spec.prompt_version != prompt_version():
        raise PromptError(f"spec {spec.node_id} was frozen under {spec.prompt_version!r}; templates on disk are "
                          f"{prompt_version()!r}")
    by_id: dict[str, tuple[GenerationSpec, Candidate | None]] = {}
    for parent_spec, candidate in parents:
        if parent_spec.node_id in by_id:
            raise PromptError(f"spec {spec.node_id}: parent {parent_spec.node_id} supplied twice")
        if candidate is not None:
            if candidate.node_id != parent_spec.node_id or candidate.candidate_id != parent_spec.node_id:
                raise PromptError(f"candidate {candidate.candidate_id} does not belong to parent {parent_spec.node_id}")
            if candidate.item_id != task.item_id:
                raise PromptError(f"candidate {candidate.candidate_id} is for item {candidate.item_id}")
        if parent_spec.item_id != task.item_id:
            raise PromptError(f"parent {parent_spec.node_id} is for item {parent_spec.item_id}")
        by_id[parent_spec.node_id] = (parent_spec, candidate)
    if set(by_id) != set(spec.parent_ids):
        raise PromptError(f"spec {spec.node_id}: parents supplied {sorted(by_id)} != declared {sorted(spec.parent_ids)}")
    return [by_id[node_id] for node_id in spec.parent_ids]


def _labelled(bodies: Sequence[str]) -> str:
    return "\n\n".join(f"{CANDIDATE_LABEL.format(index=i)}\n{body}" for i, body in enumerate(bodies, start=1))


def render_messages_with_report(spec: GenerationSpec, task: PublicTask,
                                parents: Sequence[tuple[GenerationSpec, Candidate | None]], *,
                                seed: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Messages for one node plus an audit report (packet availability, trimming, blind order, sizes)."""
    ordered = _check_request(spec, task, parents)
    template = TEMPLATE_FOR_KIND[spec.kind]
    values = {"task_prompt": task.prompt, "output_requirement": output_requirement(task)}
    packets: list[dict[str, Any]] = []
    order: list[int] | None = None

    def packet(role: str, parent: tuple[GenerationSpec, Candidate | None], label: str | None) -> str:
        parent_spec, candidate = parent
        try:
            body, entry = render_packet(candidate, source_node_id=parent_spec.node_id)
        except ContextLimitEvent as event:
            event.record["receiver_node_id"] = spec.node_id
            raise
        entry.update(role=role, label=label, source_root_slot=parent_spec.root_slot)
        packets.append(entry)
        return body

    if spec.kind in (GenerationKind.ROOT, GenerationKind.DEEP):
        if ordered:
            raise PromptError(f"{spec.kind.value} node {spec.node_id} must have no parents")
    elif spec.kind in (GenerationKind.PRIVATE, GenerationKind.CP):
        if len(ordered) != 1:
            raise PromptError(f"{spec.kind.value} node {spec.node_id} needs exactly one parent")
        values["own_packet"] = packet("own", ordered[0], None)
    elif spec.kind is GenerationKind.PEER:
        own = [p for p in ordered if p[0].root_slot == spec.root_slot]
        others = {p[0].root_slot: p for p in ordered if p[0].root_slot != spec.root_slot}
        if len(ordered) != spec.team_size or len(own) != 1 or len(others) != spec.team_size - 1:
            raise PromptError(f"peer node {spec.node_id}: parents must be the n={spec.team_size} previous-round "
                              f"nodes with distinct slots including receiver slot {spec.root_slot}")
        values["own_packet"] = packet("own", own[0], None)
        order = list(blind_order(PEER_ORDER_NAMESPACE, seed=seed, item_id=task.item_id, team_size=spec.team_size,
                                 update_round=spec.update_round, receiver_slot=spec.root_slot,
                                 source_slots=tuple(others)))
        values["peer_packets"] = _labelled([packet("peer", others[slot], CANDIDATE_LABEL.format(index=i))
                                            for i, slot in enumerate(order, start=1)])
    elif spec.kind is GenerationKind.POOLED:
        sources = {p[0].root_slot: p for p in ordered}
        if not ordered or len(sources) != len(ordered):
            raise PromptError(f"pooled node {spec.node_id}: needs parents with distinct slots")
        order = list(blind_order(POOLED_ORDER_NAMESPACE, seed=seed, item_id=task.item_id, team_size=spec.team_size,
                                 update_round=spec.update_round, receiver_slot=spec.root_slot,
                                 source_slots=tuple(sources)))
        values["candidate_packets"] = _labelled([packet("pooled", sources[slot], CANDIDATE_LABEL.format(index=i))
                                                 for i, slot in enumerate(order, start=1)])
    else:  # pragma: no cover - GenerationKind is closed
        raise PromptError(f"unsupported kind {spec.kind!r}")

    messages = [{"role": "system", "content": _template_text("system.txt")},
                {"role": "user", "content": _substitute(template, values)}]
    report: dict[str, Any] = {
        "node_id": spec.node_id,
        "kind": spec.kind.value,
        "template": template,
        "prompt_version": prompt_version(),
        "envelope_chars": PACKET_ENVELOPE_CHARS,
        "display_order_slots": order,
        "packets": packets,
        "trimmed_packets": sum(1 for p in packets if p["trimmed"]),
        "chars_removed_total": sum(p["chars_removed"] for p in packets),
        "bytes_removed_total": sum(p["bytes_removed"] for p in packets),
        "system_chars": len(messages[0]["content"]),
        "user_chars": len(messages[1]["content"]),
        "rendered_chars": sum(len(m["content"]) for m in messages),
        "messages_sha256": _audit_hash(messages),      # audit only, never a cache key (R5)
    }
    return messages, report


def render_messages(spec: GenerationSpec, task: PublicTask,
                    parents: Sequence[tuple[GenerationSpec, Candidate | None]], *, seed: int) -> list[dict[str, str]]:
    return render_messages_with_report(spec, task, parents, seed=seed)[0]
